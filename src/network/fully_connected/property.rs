
use ndarray::{Array1, Array2};
use crate::network::FullyConnected;

use super::fully_connected::Activation;


impl FullyConnected {
    pub fn new(input_size: usize, output_size: usize, activation: Option<Activation>) -> Self {
        FullyConnected {
            weight: Array2::zeros((input_size, output_size)),
            bias: Array1::zeros(output_size),
            forward: None,
            activated: None,
            activation,
            gradient: None,
        }
    }

    pub fn shape(&self) -> &[usize] {
        let shape = self.weight.shape();
        shape
    }

    pub fn get_linear_output(&self) -> Array2<f32> {
        match &self.activated {
            Some(activ) => activ.clone(),
            None => {
                match &self.forward {
                    Some(forward) => forward.clone(),
                    None => panic!("no linear output exist!"),
                }
            }
        }
    }
}
