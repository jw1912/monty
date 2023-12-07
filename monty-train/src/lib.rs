mod data;
mod datagen;
mod gradient;
mod rng;

pub use data::TrainingPosition;
pub use datagen::{set_stop, write_data, DatagenThread};
pub use gradient::gradient_batch;
pub use rng::Rand;

pub fn to_slice_with_lifetime<T, U>(slice: &[T]) -> &[U] {
    let src_size = std::mem::size_of_val(slice);
    let tgt_size = std::mem::size_of::<U>();

    assert!(
        src_size % tgt_size == 0,
        "Target type size does not divide slice size!"
    );

    let len = src_size / tgt_size;
    unsafe { std::slice::from_raw_parts(slice.as_ptr().cast(), len) }
}
