
use ndarray::Array2;
use crate::network::{Activation, FullyConnected};

use super::fully_connected::relu;


impl FullyConnected {
    pub fn backward(&mut self, delta: &Array2<f32>, weight: &Array2<f32>) -> Array2<f32> {
        let z = match &self.forward {
            Some(forward) => forward,
            None => panic!("not forwarded values to calculate backpropagation"),
        };
        
        let grad = delta.dot(&weight.t()) * relu(z.clone(), true);
        self.gradient = Some(grad.clone());
        grad
    }
}
