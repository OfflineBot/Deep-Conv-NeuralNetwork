use ndarray::Array4;
use crate::network::ConvLayer;



impl ConvLayer {
    pub fn forward(&mut self, x: Array4<f32>) {
        
    }

    pub fn get_forwarded_output(&self) -> Array4<f32> {
        match &self.forwarded {
            Some(value) => value.clone(),
            None => panic!("No forwarded input give in convolutional layer"),
        }
    }
}
