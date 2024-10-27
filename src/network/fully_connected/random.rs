use ndarray::{Array1, Array2};
use crate::network::FullyConnected;
use ndarray_rand::{rand_distr::Uniform, RandomExt};


impl FullyConnected {
    pub fn fill_random(&mut self, minimum: f32, maximum: f32) {
        let distr = Uniform::new(minimum, maximum);
        let size = self.weight.shape();

        let new_weigth = Array2::random((size[0], size[1]), distr);
        let new_bias = Array1::random(size[1], distr);

        self.weight = new_weigth;
        self.bias = new_bias;
    }
}
