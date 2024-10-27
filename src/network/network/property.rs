
use super::network::Network;

impl Network {
    pub fn new() -> Self {
        Network {
            convolutions: Vec::new(),
            fully_connected: Vec::new(),
        }
    }
}
