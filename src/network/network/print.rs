use crate::network::{Activation, Network};


impl Network {
    pub fn print_network(&self) {
        println!(" ==== Network Layout ====");
        if self.convolutions.len() > 0 {
            println!(" --- Convolutional Layer ---");
            for i in self.convolutions.iter() {
                println!("{} - {}", i.in_channel, i.out_channel);
            }
        }

        if self.fully_connected.len() > 0 {
            println!(" --- Fully Connected Layer ---");
            for i in self.fully_connected.iter() {
                println!("{} - {} | [{:?}]", i.shape()[0], i.shape()[1], i.activation);
            }
        }
        println!(" ====    ========    ====");
    }
}
