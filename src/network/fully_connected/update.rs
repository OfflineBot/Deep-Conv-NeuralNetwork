use ndarray::Array2;

use crate::network::FullyConnected;


impl FullyConnected {
    // weight -= weight - (learning_rate * input.t().dot(delta))
    pub fn update(&mut self, learning_rate: f32, input: &Array2<f32>) {
        let delta: &Array2<f32> = match &self.gradient {
            Some(grad) => grad,
            None => panic!("No Gradients from backpropagation available"),
        };

        self.weight = &self.weight - (learning_rate * input.t().dot(delta));
        self.bias = &self.bias - (learning_rate * delta.sum_axis(ndarray::Axis(0)));
    }
}
