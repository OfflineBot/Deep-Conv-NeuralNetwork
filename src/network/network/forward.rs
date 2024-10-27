use ndarray::{Array2, Array4};

use crate::network::Network;


impl Network {

    pub fn forward_conv(&mut self, data: Array4<f32>) -> Array4<f32> {
        let mut input_data = data;
        for i in self.convolutions.iter_mut() {
            i.forward(input_data);
            input_data = i.get_forwarded_output();
        }
        input_data
    }

    pub fn flatten_conv_output_for_fc(&self) -> Array2<f32> {
        let index = self.convolutions.len() - 1;
        let data = match &self.convolutions[index].forwarded {
            Some(value) => value.clone(),
            None => panic!("There is not convoluted output"),
        };

        let (n, c, h, w) = data.dim();
        data.into_shape_with_order((n, c * h * w)).unwrap()
    }


    pub fn forward_fc(&mut self, data: Array2<f32>) -> Array2<f32> {
        let mut input_data = data;
        for i in self.fully_connected.iter_mut() {
            i.forward(&input_data);
            input_data = i.get_linear_output().clone();
        }
        input_data
    }
}
