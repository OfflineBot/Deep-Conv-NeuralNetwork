use ndarray::Array2;

use crate::network::{Activation, FullyConnected};

use super::fully_connected::relu;




impl FullyConnected {
    pub fn forward(&mut self, x: &Array2<f32>) {
        let z1 = &x.dot(&self.weight) + &self.bias;
        self.forward = Some(z1.clone());
        match &self.activation {
            Some(activatio) => {
                match activatio {
                    Activation::ReLU => self.activated = Some(relu(z1, false)),
                    Activation::Sigmoid => panic!("sigmoid not implemented yet"),
                }
            },
            None => {}
        }
    }
}

