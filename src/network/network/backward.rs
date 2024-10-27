
use ndarray::Array2;
use crate::network::Network;


impl Network {
    pub fn backward(&mut self, truth: &Array2<f32>, prediction: &Array2<f32>) {
        let mut delta = prediction - truth;
        let fully_connected_len = self.fully_connected.len();
        let mut weight = self.fully_connected[fully_connected_len-1].weight.clone();
        self.fully_connected[fully_connected_len-1].gradient = Some(delta.clone());

        for i in self.fully_connected.iter_mut().rev().skip(1) {
            i.backward(&delta, &weight);
            weight = i.weight.clone();
            delta = i.gradient.clone().unwrap();
        }
    }
}

