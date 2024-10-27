
use crate::network::Network;
use std::io::{Error, ErrorKind};


impl Network {

    fn validate_conv(&self, image_shape: Option<(usize, usize)>) -> usize {

        let img_shape = match image_shape {
            Some(shape) => shape,
            None => panic!("No image shape were given!"),
        };
        let mut img_height_out: usize = 0;
        let mut img_width_out: usize = 0;
        let mut img_height_in = img_shape.0;
        let mut img_width_in = img_shape.1;

        for i in self.convolutions.iter() {
            img_height_out = ((img_height_in - i.kernel_size + 2 * i.padding) / i.stride) + 1;
            img_width_out = ((img_width_in - i.kernel_size + 2 * i.padding) / i.stride) + 1;
            img_height_in = img_height_out;
            img_width_in = img_height_out;
        }

        let last_out_channel = self.convolutions[self.convolutions.len()-1].out_channel;
        &img_width_out * &img_height_out * &last_out_channel
    }

    fn validate_fc(&self, mut output_match: [usize; 2]) -> bool {
        let mut has_error = false;
        if output_match[0] == 0 {
            let first_shape = self.fully_connected[0].shape();
            if output_match[1] == first_shape[0] {
                println!("Nx{} * {}x{}", output_match[1], first_shape[0], first_shape[1]);
            } else {
                println!("Nx{} * {}x{} | does not match", output_match[1], first_shape[0], first_shape[1]);
                has_error = true;
            }
            output_match = [first_shape[0], first_shape[1]];
        }
        
        for (idx, i) in self.fully_connected.iter().enumerate() {
            if idx == 0 {
                continue;
            }

            if output_match[1] == i.shape()[0] {
                println!("{}x{} * {}x{}", output_match[0], output_match[1], i.shape()[0], i.shape()[1]);
            } else {
                println!("{}x{} * {}x{} | does not match", output_match[0], output_match[1], i.shape()[0], i.shape()[1]);
                has_error = true;
            }

            output_match = [i.shape()[0], i.shape()[1]];
        }

        return has_error;
    }

    pub fn validate_sizes(&self, image_shape: Option<(usize, usize)>) -> std::io::Result<()> {
        let mut output_match: [usize; 2] = [0, 0];
        let mut found_error = false;

        if self.fully_connected.len() == 0 && self.convolutions.len() == 0 {
            println!("> no layers found at all");
            return Ok(());
        } else if self.fully_connected.len() == 0 && self.convolutions.len() > 0 {
            println!("> only found convolutional layer");
            self.validate_conv(image_shape);
        } else if self.fully_connected.len() > 0 && self.convolutions.len() == 0 {
            println!("> only found fully connected layer");
            let fully_connected_shape = self.fully_connected[0].shape();
            output_match = [fully_connected_shape[0], fully_connected_shape[1]];
            found_error = self.validate_fc(output_match);
        } else {
            output_match = [0, self.validate_conv(image_shape)];
            found_error = self.validate_fc(output_match);
        }

        match found_error {
            true => {
                println!("---\nnot all layer sizes match!");
                Err(Error::new(ErrorKind::Other, "Layer Sizes dont match!"))
            }
            false => {
                println!("all sizes match!");
                Ok(())
            }
        }
    }

}
