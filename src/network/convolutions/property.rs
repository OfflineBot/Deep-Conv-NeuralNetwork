use ndarray::Array3;

use crate::network::ConvLayer;


impl ConvLayer {
    pub fn new(in_channel: usize, out_channel: usize, kernel_size: usize, padding: usize, stride: usize) -> Self {
        ConvLayer {
            in_channel,
            out_channel,
            kernel_size,
            kernel: Array3::zeros((out_channel, kernel_size, kernel_size)),
            padding,
            stride,
            forwarded: None,
        }
    }
}
