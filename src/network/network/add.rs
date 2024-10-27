use crate::network::Activation;
use crate::network::ConvLayer;
use crate::network::Network;
use crate::network::FullyConnected;


impl Network {
    pub fn add_fc(&mut self, input_size: usize, output_size: usize, activation: Option<Activation>) {
        self.fully_connected.push(
            FullyConnected::new(input_size, output_size, activation)
        );
    }

    pub fn add_conv(&mut self, in_channel: usize, out_channel: usize, kernel_size: usize, padding: usize, stride: usize) {
        self.convolutions.push(
            ConvLayer::new(in_channel, out_channel, kernel_size, padding, stride)
        );
    }
}
