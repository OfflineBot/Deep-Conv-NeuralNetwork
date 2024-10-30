use ndarray::Array3;
use ndarray_rand::{rand_distr::Uniform, RandomExt};

use crate::network::ConvLayer;


impl ConvLayer {
    pub fn fill_random(&mut self, minimum: f32, maximum: f32) {
        let shape = self.kernel.shape();
        let distr = Uniform::new(minimum, maximum);

        self.kernel = Array3::random((shape[0], shape[1], shape[2]), distr);
    }
}
