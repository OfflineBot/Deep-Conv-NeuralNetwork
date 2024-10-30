
use super::network::Network;

impl Network {
    // randomizes weights and kernels
    pub fn randomize(&mut self, minimum: f32, maximum: f32) {

        for i in self.convolutions.iter_mut() {
            i.fill_random(minimum, maximum);
        }

        for i in self.fully_connected.iter_mut() {
            i.fill_random(minimum, maximum);
        }

    }
}
