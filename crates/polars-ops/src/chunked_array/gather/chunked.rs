use std::borrow::Cow;
use std::fmt::Debug;

use arrow::array::{Array, BinaryViewArrayGeneric, View, ViewType};
use arrow::bitmap::BitmapBuilder;
use arrow::buffer::Buffer;
use arrow::legacy::trusted_len::TrustedLenPush;
use hashbrown::hash_map::Entry;
use polars_core::prelude::gather::_update_gather_sorted_flag;
use polars_core::prelude::*;
use polars_core::series::IsSorted;
use polars_core::utils::Container;
use polars_core::with_match_physical_numeric_polars_type;

use crate::frame::IntoDf;

/// Gather by [`ChunkId`]
pub trait TakeChunked {
    /// # Safety
    /// This function doesn't do any bound checks.
    unsafe fn take_chunked_unchecked<const B: u64>(
        &self,
        by: &[ChunkId<B>],
        sorted: IsSorted,
    ) -> Self;

    /// # Safety
    /// This function doesn't do any bound checks.
    unsafe fn take_opt_chunked_unchecked<const B: u64>(&self, by: &[ChunkId<B>]) -> Self;
}

impl TakeChunked for DataFrame {
    /// Take elements by a slice of [`ChunkId`]s.
    ///
    /// # Safety
    /// Does not do any bound checks.
    /// `sorted` indicates if the chunks are sorted.
    unsafe fn take_chunked_unchecked<const B: u64>(
        &self,
        idx: &[ChunkId<B>],
        sorted: IsSorted,
    ) -> DataFrame {
        let cols = self
            .to_df()
            ._apply_columns(&|s| s.take_chunked_unchecked(idx, sorted));

        unsafe { DataFrame::new_no_checks_height_from_first(cols) }
    }

    /// Take elements by a slice of optional [`ChunkId`]s.
    ///
    /// # Safety
    /// Does not do any bound checks.
    unsafe fn take_opt_chunked_unchecked<const B: u64>(&self, idx: &[ChunkId<B>]) -> DataFrame {
        let cols = self
            .to_df()
            ._apply_columns(&|s| s.take_opt_chunked_unchecked(idx));

        unsafe { DataFrame::new_no_checks_height_from_first(cols) }
    }
}

pub trait TakeChunkedHorPar: IntoDf {
    /// # Safety
    /// Doesn't perform any bound checks
    unsafe fn _take_chunked_unchecked_hor_par<const B: u64>(
        &self,
        idx: &[ChunkId<B>],
        sorted: IsSorted,
    ) -> DataFrame {
        let cols = self
            .to_df()
            ._apply_columns_par(&|s| s.take_chunked_unchecked(idx, sorted));

        unsafe { DataFrame::new_no_checks_height_from_first(cols) }
    }

    /// # Safety
    /// Doesn't perform any bound checks
    ///
    /// Check for null state in `ChunkId`.
    unsafe fn _take_opt_chunked_unchecked_hor_par<const B: u64>(
        &self,
        idx: &[ChunkId<B>],
    ) -> DataFrame {
        let cols = self
            .to_df()
            ._apply_columns_par(&|s| s.take_opt_chunked_unchecked(idx));

        unsafe { DataFrame::new_no_checks_height_from_first(cols) }
    }
}

impl TakeChunkedHorPar for DataFrame {}

fn prepare_series(s: &Series) -> Cow<Series> {
    let phys = if s.dtype().is_nested() {
        Cow::Borrowed(s)
    } else {
        s.to_physical_repr()
    };
    // If this is hit the cast rechunked the data and the gather will OOB
    assert_eq!(
        phys.chunks().len(),
        s.chunks().len(),
        "implementation error"
    );
    phys
}

impl TakeChunked for Column {
    unsafe fn take_chunked_unchecked<const B: u64>(
        &self,
        by: &[ChunkId<B>],
        sorted: IsSorted,
    ) -> Self {
        // @scalar-opt
        let s = self.as_materialized_series();
        let s = unsafe { s.take_chunked_unchecked(by, sorted) };
        s.into_column()
    }

    unsafe fn take_opt_chunked_unchecked<const B: u64>(&self, by: &[ChunkId<B>]) -> Self {
        // @scalar-opt
        let s = self.as_materialized_series();
        let s = unsafe { s.take_opt_chunked_unchecked(by) };
        s.into_column()
    }
}

impl TakeChunked for Series {
    unsafe fn take_chunked_unchecked<const B: u64>(
        &self,
        by: &[ChunkId<B>],
        sorted: IsSorted,
    ) -> Self {
        let phys = prepare_series(self);
        use DataType::*;
        let out = match phys.dtype() {
            dt if dt.is_numeric() => {
                with_match_physical_numeric_polars_type!(phys.dtype(), |$T| {
                 let ca: &ChunkedArray<$T> = phys.as_ref().as_ref().as_ref();
                 ca.take_chunked_unchecked(by, sorted).into_series()
                })
            },
            Boolean => {
                let ca = phys.bool().unwrap();
                ca.take_chunked_unchecked(by, sorted).into_series()
            },
            Binary => {
                let ca = phys.binary().unwrap();
                take_unchecked_binview(ca, by, sorted).into_series()
            },
            String => {
                let ca = phys.str().unwrap();
                take_unchecked_binview(ca, by, sorted).into_series()
            },
            List(_) => {
                let ca = phys.list().unwrap();
                ca.take_chunked_unchecked(by, sorted).into_series()
            },
            #[cfg(feature = "dtype-array")]
            Array(_, _) => {
                let ca = phys.array().unwrap();
                ca.take_chunked_unchecked(by, sorted).into_series()
            },
            #[cfg(feature = "dtype-struct")]
            Struct(_) => {
                let ca = phys.struct_().unwrap();
                ca._apply_fields(|s| s.take_chunked_unchecked(by, sorted))
                    .expect("infallible")
                    .into_series()
            },
            #[cfg(feature = "object")]
            Object(_, _) => take_unchecked_object(&phys, by, sorted),
            #[cfg(feature = "dtype-decimal")]
            Decimal(_, _) => {
                let ca = phys.decimal().unwrap();
                let out = ca.0.take_chunked_unchecked(by, sorted);
                out.into_decimal_unchecked(ca.precision(), ca.scale())
                    .into_series()
            },
            Null => Series::new_null(self.name().clone(), by.len()),
            _ => unreachable!(),
        };
        unsafe { out.from_physical_unchecked(self.dtype()).unwrap() }
    }

    /// Take function that checks of null state in `ChunkIdx`.
    unsafe fn take_opt_chunked_unchecked<const B: u64>(&self, by: &[ChunkId<B>]) -> Self {
        let phys = prepare_series(self);
        use DataType::*;
        let out = match phys.dtype() {
            dt if dt.is_numeric() => {
                with_match_physical_numeric_polars_type!(phys.dtype(), |$T| {
                 let ca: &ChunkedArray<$T> = phys.as_ref().as_ref().as_ref();
                 ca.take_opt_chunked_unchecked(by).into_series()
                })
            },
            Boolean => {
                let ca = phys.bool().unwrap();
                ca.take_opt_chunked_unchecked(by).into_series()
            },
            Binary => {
                let ca = phys.binary().unwrap();
                take_unchecked_binview_opt(ca, by).into_series()
            },
            String => {
                let ca = phys.str().unwrap();
                take_unchecked_binview_opt(ca, by).into_series()
            },
            List(_) => {
                let ca = phys.list().unwrap();
                ca.take_opt_chunked_unchecked(by).into_series()
            },
            #[cfg(feature = "dtype-array")]
            Array(_, _) => {
                let ca = phys.array().unwrap();
                ca.take_opt_chunked_unchecked(by).into_series()
            },
            #[cfg(feature = "dtype-struct")]
            Struct(_) => {
                let ca = phys.struct_().unwrap();
                ca._apply_fields(|s| s.take_opt_chunked_unchecked(by))
                    .expect("infallible")
                    .into_series()
            },
            #[cfg(feature = "object")]
            Object(_, _) => take_opt_unchecked_object(&phys, by),
            #[cfg(feature = "dtype-decimal")]
            Decimal(_, _) => {
                let ca = phys.decimal().unwrap();
                let out = ca.0.take_opt_chunked_unchecked(by);
                out.into_decimal_unchecked(ca.precision(), ca.scale())
                    .into_series()
            },
            Null => Series::new_null(self.name().clone(), by.len()),
            _ => unreachable!(),
        };
        unsafe { out.from_physical_unchecked(self.dtype()).unwrap() }
    }
}

impl<T> TakeChunked for ChunkedArray<T>
where
    T: PolarsDataType,
    T::Array: Debug,
{
    unsafe fn take_chunked_unchecked<const B: u64>(
        &self,
        by: &[ChunkId<B>],
        sorted: IsSorted,
    ) -> Self {
        let arrow_dtype = self.dtype().to_arrow(CompatLevel::newest());

        let mut out = if let Some(iter) = self.downcast_slices() {
            let targets = iter.collect::<Vec<_>>();
            let iter = by.iter().map(|chunk_id| {
                debug_assert!(
                    !chunk_id.is_null(),
                    "null chunks should not hit this branch"
                );
                let (chunk_idx, array_idx) = chunk_id.extract();
                let vals = targets.get_unchecked(chunk_idx as usize);
                vals.get_unchecked(array_idx as usize).clone()
            });

            let arr = iter.collect_arr_trusted_with_dtype(arrow_dtype);
            ChunkedArray::with_chunk(self.name().clone(), arr)
        } else {
            let targets = self.downcast_iter().collect::<Vec<_>>();
            let iter = by.iter().map(|chunk_id| {
                debug_assert!(
                    !chunk_id.is_null(),
                    "null chunks should not hit this branch"
                );
                let (chunk_idx, array_idx) = chunk_id.extract();
                let vals = targets.get_unchecked(chunk_idx as usize);
                vals.get_unchecked(array_idx as usize)
            });
            let arr = iter.collect_arr_trusted_with_dtype(arrow_dtype);
            ChunkedArray::with_chunk(self.name().clone(), arr)
        };
        let sorted_flag = _update_gather_sorted_flag(self.is_sorted_flag(), sorted);
        out.set_sorted_flag(sorted_flag);
        out
    }

    // Take function that checks of null state in `ChunkIdx`.
    unsafe fn take_opt_chunked_unchecked<const B: u64>(&self, by: &[ChunkId<B>]) -> Self {
        let arrow_dtype = self.dtype().to_arrow(CompatLevel::newest());

        if let Some(iter) = self.downcast_slices() {
            let targets = iter.collect::<Vec<_>>();
            let arr = by
                .iter()
                .map(|chunk_id| {
                    if chunk_id.is_null() {
                        None
                    } else {
                        let (chunk_idx, array_idx) = chunk_id.extract();
                        let vals = *targets.get_unchecked(chunk_idx as usize);
                        Some(vals.get_unchecked(array_idx as usize).clone())
                    }
                })
                .collect_arr_trusted_with_dtype(arrow_dtype);

            ChunkedArray::with_chunk(self.name().clone(), arr)
        } else {
            let targets = self.downcast_iter().collect::<Vec<_>>();
            let arr = by
                .iter()
                .map(|chunk_id| {
                    if chunk_id.is_null() {
                        None
                    } else {
                        let (chunk_idx, array_idx) = chunk_id.extract();
                        let vals = *targets.get_unchecked(chunk_idx as usize);
                        vals.get_unchecked(array_idx as usize)
                    }
                })
                .collect_arr_trusted_with_dtype(arrow_dtype);

            ChunkedArray::with_chunk(self.name().clone(), arr)
        }
    }
}

#[cfg(feature = "object")]
unsafe fn take_unchecked_object<const B: u64>(
    s: &Series,
    by: &[ChunkId<B>],
    _sorted: IsSorted,
) -> Series {
    let DataType::Object(_, reg) = s.dtype() else {
        unreachable!()
    };
    let reg = reg.as_ref().unwrap();
    let mut builder = (*reg.builder_constructor)(s.name().clone(), by.len());

    by.iter().for_each(|chunk_id| {
        let (chunk_idx, array_idx) = chunk_id.extract();
        let object = s.get_object_chunked_unchecked(chunk_idx as usize, array_idx as usize);
        builder.append_option(object.map(|v| v.as_any()))
    });
    builder.to_series()
}

#[cfg(feature = "object")]
unsafe fn take_opt_unchecked_object<const B: u64>(s: &Series, by: &[ChunkId<B>]) -> Series {
    let DataType::Object(_, reg) = s.dtype() else {
        unreachable!()
    };
    let reg = reg.as_ref().unwrap();
    let mut builder = (*reg.builder_constructor)(s.name().clone(), by.len());

    by.iter().for_each(|chunk_id| {
        if chunk_id.is_null() {
            builder.append_null()
        } else {
            let (chunk_idx, array_idx) = chunk_id.extract();
            let object = s.get_object_chunked_unchecked(chunk_idx as usize, array_idx as usize);
            builder.append_option(object.map(|v| v.as_any()))
        }
    });
    builder.to_series()
}

unsafe fn update_view(
    mut view: View,
    orig_buffers: &[Buffer<u8>],
    buffer_idxs: &mut PlHashMap<(*const u8, usize), u32>,
    buffers: &mut Vec<Buffer<u8>>,
) -> View {
    if view.length > 12 {
        // Dedup on pointer + length.
        let orig_buffer = orig_buffers.get_unchecked(view.buffer_idx as usize);
        view.buffer_idx =
            match buffer_idxs.entry((orig_buffer.as_slice().as_ptr(), orig_buffer.len())) {
                Entry::Occupied(o) => *o.get(),
                Entry::Vacant(v) => {
                    let buffer_idx = buffers.len() as u32;
                    buffers.push(orig_buffer.clone());
                    v.insert(buffer_idx);
                    buffer_idx
                },
            };
    }
    view
}

unsafe fn take_unchecked_binview<const B: u64, T, V>(
    ca: &ChunkedArray<T>,
    by: &[ChunkId<B>],
    sorted: IsSorted,
) -> ChunkedArray<T>
where
    T: PolarsDataType<Array = BinaryViewArrayGeneric<V>>,
    V: ViewType + ?Sized,
{
    let mut views = Vec::with_capacity(by.len());
    let (validity, arc_data_buffers);

    // If we can cheaply clone the list of buffers from the ChunkedArray we will,
    // otherwise we will only clone those buffers we need.
    if ca.n_chunks() == 1 {
        let arr = ca.downcast_iter().next().unwrap();
        let arr_views = arr.views();

        validity = if arr.has_nulls() {
            let mut validity = BitmapBuilder::with_capacity(by.len());
            for id in by.iter() {
                let (chunk_idx, array_idx) = id.extract();
                debug_assert!(chunk_idx == 0);
                if arr.is_null_unchecked(array_idx as usize) {
                    views.push_unchecked(View::default());
                    validity.push_unchecked(false);
                } else {
                    views.push_unchecked(*arr_views.get_unchecked(array_idx as usize));
                    validity.push_unchecked(true);
                }
            }
            Some(validity.freeze())
        } else {
            for id in by.iter() {
                let (chunk_idx, array_idx) = id.extract();
                debug_assert!(chunk_idx == 0);
                views.push_unchecked(*arr_views.get_unchecked(array_idx as usize));
            }
            None
        };

        arc_data_buffers = arr.data_buffers().clone();
    } else {
        let mut buffer_idxs = PlHashMap::with_capacity(8);
        let mut buffers = Vec::with_capacity(8);

        validity = if ca.has_nulls() {
            let mut validity = BitmapBuilder::with_capacity(by.len());
            for id in by.iter() {
                let (chunk_idx, array_idx) = id.extract();

                let arr = ca.downcast_get_unchecked(chunk_idx as usize);
                if arr.is_null_unchecked(array_idx as usize) {
                    views.push_unchecked(View::default());
                    validity.push_unchecked(false);
                } else {
                    let view = *arr.views().get_unchecked(array_idx as usize);
                    views.push_unchecked(update_view(
                        view,
                        arr.data_buffers(),
                        &mut buffer_idxs,
                        &mut buffers,
                    ));
                    validity.push_unchecked(true);
                }
            }
            Some(validity.freeze())
        } else {
            for id in by.iter() {
                let (chunk_idx, array_idx) = id.extract();

                let arr = ca.downcast_get_unchecked(chunk_idx as usize);
                let view = *arr.views().get_unchecked(array_idx as usize);
                views.push_unchecked(update_view(
                    view,
                    arr.data_buffers(),
                    &mut buffer_idxs,
                    &mut buffers,
                ));
            }
            None
        };

        arc_data_buffers = buffers.into();
    };

    let arr = BinaryViewArrayGeneric::<V>::new_unchecked_unknown_md(
        V::DATA_TYPE,
        views.into(),
        arc_data_buffers,
        validity,
        None,
    );

    let mut out = ChunkedArray::with_chunk(ca.name().clone(), arr.maybe_gc());
    let sorted_flag = _update_gather_sorted_flag(ca.is_sorted_flag(), sorted);
    out.set_sorted_flag(sorted_flag);
    out
}

unsafe fn take_unchecked_binview_opt<const B: u64, T, V>(
    ca: &ChunkedArray<T>,
    by: &[ChunkId<B>],
) -> ChunkedArray<T>
where
    T: PolarsDataType<Array = BinaryViewArrayGeneric<V>>,
    V: ViewType + ?Sized,
{
    let mut views = Vec::with_capacity(by.len());
    let mut validity = BitmapBuilder::with_capacity(by.len());

    // If we can cheaply clone the list of buffers from the ChunkedArray we will,
    // otherwise we will only clone those buffers we need.
    let arc_data_buffers = if ca.n_chunks() == 1 {
        let arr = ca.downcast_iter().next().unwrap();
        let arr_views = arr.views();

        if arr.has_nulls() {
            for id in by.iter() {
                let (chunk_idx, array_idx) = id.extract();
                debug_assert!(id.is_null() || chunk_idx == 0);
                if id.is_null() || arr.is_null_unchecked(array_idx as usize) {
                    views.push_unchecked(View::default());
                    validity.push_unchecked(false);
                } else {
                    views.push_unchecked(*arr_views.get_unchecked(array_idx as usize));
                    validity.push_unchecked(true);
                }
            }
        } else {
            for id in by.iter() {
                let (chunk_idx, array_idx) = id.extract();
                debug_assert!(id.is_null() || chunk_idx == 0);
                if id.is_null() {
                    views.push_unchecked(View::default());
                    validity.push_unchecked(false);
                } else {
                    views.push_unchecked(*arr_views.get_unchecked(array_idx as usize));
                    validity.push_unchecked(true);
                }
            }
        }

        arr.data_buffers().clone()
    } else {
        let mut buffer_idxs = PlHashMap::with_capacity(8);
        let mut buffers = Vec::with_capacity(8);

        if ca.has_nulls() {
            for id in by.iter() {
                let (chunk_idx, array_idx) = id.extract();

                if id.is_null() {
                    views.push_unchecked(View::default());
                    validity.push_unchecked(false);
                } else {
                    let arr = ca.downcast_get_unchecked(chunk_idx as usize);
                    if arr.is_null_unchecked(array_idx as usize) {
                        views.push_unchecked(View::default());
                        validity.push_unchecked(false);
                    } else {
                        let view = *arr.views().get_unchecked(array_idx as usize);
                        views.push_unchecked(update_view(
                            view,
                            arr.data_buffers(),
                            &mut buffer_idxs,
                            &mut buffers,
                        ));
                        validity.push_unchecked(true);
                    }
                }
            }
        } else {
            for id in by.iter() {
                let (chunk_idx, array_idx) = id.extract();

                if id.is_null() {
                    views.push_unchecked(View::default());
                    validity.push_unchecked(false);
                } else {
                    let arr = ca.downcast_get_unchecked(chunk_idx as usize);
                    let view = *arr.views().get_unchecked(array_idx as usize);
                    views.push_unchecked(update_view(
                        view,
                        arr.data_buffers(),
                        &mut buffer_idxs,
                        &mut buffers,
                    ));
                    validity.push_unchecked(true);
                }
            }
        };

        buffers.into()
    };

    let arr = BinaryViewArrayGeneric::<V>::new_unchecked_unknown_md(
        V::DATA_TYPE,
        views.into(),
        arc_data_buffers,
        Some(validity.freeze()),
        None,
    );

    ChunkedArray::with_chunk(ca.name().clone(), arr.maybe_gc())
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_binview_chunked_gather() {
        unsafe {
            // # Series without nulls;
            let mut s_1 = Series::new(
                "a".into(),
                &["1 loooooooooooong string", "2 loooooooooooong string"],
            );
            let s_2 = Series::new(
                "a".into(),
                &["11 loooooooooooong string", "22 loooooooooooong string"],
            );
            let s_3 = Series::new(
                "a".into(),
                &[
                    "111 loooooooooooong string",
                    "222 loooooooooooong string",
                    "small", // this tests we don't mess with the inlined view
                ],
            );
            s_1.append(&s_2).unwrap();
            s_1.append(&s_3).unwrap();

            assert_eq!(s_1.n_chunks(), 3);

            // ## Ids without nulls;
            let by: [ChunkId<24>; 7] = [
                ChunkId::store(0, 0),
                ChunkId::store(0, 1),
                ChunkId::store(1, 1),
                ChunkId::store(1, 0),
                ChunkId::store(2, 0),
                ChunkId::store(2, 1),
                ChunkId::store(2, 2),
            ];

            let out = s_1.take_chunked_unchecked(&by, IsSorted::Not);
            let idx = IdxCa::new("".into(), [0, 1, 3, 2, 4, 5, 6]);
            let expected = s_1.rechunk().take(&idx).unwrap();
            assert!(out.equals(&expected));

            // ## Ids with nulls;
            let by: [ChunkId<24>; 4] = [
                ChunkId::null(),
                ChunkId::store(0, 1),
                ChunkId::store(1, 1),
                ChunkId::store(1, 0),
            ];
            let out = s_1.take_opt_chunked_unchecked(&by);

            let idx = IdxCa::new("".into(), [None, Some(1), Some(3), Some(2)]);
            let expected = s_1.rechunk().take(&idx).unwrap();
            assert!(out.equals_missing(&expected));

            // # Series with nulls;
            let mut s_1 = Series::new(
                "a".into(),
                &["1 loooooooooooong string 1", "2 loooooooooooong string 2"],
            );
            let s_2 = Series::new("a".into(), &[Some("11 loooooooooooong string 11"), None]);
            s_1.append(&s_2).unwrap();

            // ## Ids without nulls;
            let by: [ChunkId<24>; 4] = [
                ChunkId::store(0, 0),
                ChunkId::store(0, 1),
                ChunkId::store(1, 1),
                ChunkId::store(1, 0),
            ];

            let out = s_1.take_chunked_unchecked(&by, IsSorted::Not);
            let idx = IdxCa::new("".into(), [0, 1, 3, 2]);
            let expected = s_1.rechunk().take(&idx).unwrap();
            assert!(out.equals_missing(&expected));

            // ## Ids with nulls;
            let by: [ChunkId<24>; 4] = [
                ChunkId::null(),
                ChunkId::store(0, 1),
                ChunkId::store(1, 1),
                ChunkId::store(1, 0),
            ];
            let out = s_1.take_opt_chunked_unchecked(&by);

            let idx = IdxCa::new("".into(), [None, Some(1), Some(3), Some(2)]);
            let expected = s_1.rechunk().take(&idx).unwrap();
            assert!(out.equals_missing(&expected));
        }
    }
}
