
use ndarray::Array2;
use crate::network::Network;


impl Network {
    pub fn update(&mut self, learning_rate: f32, input_data: &Array2<f32>) {

        self.fully_connected[0].update(learning_rate, input_data);
        let mut input = match &self.fully_connected[0].activated {
            Some(activ) => activ.clone(),
            None => panic!("Missing forward values (a)"),
        };

        for i in self.fully_connected.iter_mut().skip(1) {
            i.update(learning_rate, &input);
            input = match &i.activated {
                Some(val) => val.clone(),
                None => input,
            };
        }
    }
}
