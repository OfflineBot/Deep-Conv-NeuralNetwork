
use ndarray::{Array3, Array4};

pub struct ConvLayer {
    pub(crate) in_channel: usize,
    pub(crate) out_channel: usize,
    pub(crate) kernel: Array3<f32>,
    pub(crate) kernel_size: usize,
    pub(crate) padding: usize,
    pub(crate) stride: usize,
    pub(crate) forwarded: Option<Array4<f32>>,
}

