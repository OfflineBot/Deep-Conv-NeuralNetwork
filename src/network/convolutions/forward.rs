use ndarray::Array4;
use crate::network::ConvLayer;



impl ConvLayer {

    fn add_padding(x: &Array4<f32>, padding: usize) -> Array4<f32> {
        let x_shape = x.shape();
        let mut output = Array4::zeros((x_shape[0], x_shape[1], x_shape[2] + (padding*2), x_shape[3] + (padding*2)));

        for size in 0..x_shape[0] {
            for channel in 0..x_shape[1] {
                for i in 0..x_shape[2] {
                    for j in 0..x_shape[3] {
                        output[[size, channel, i+padding, j+padding]] = x[[size, channel, i, j]];
                    }
                }
            }
        }

        output
    }

    pub fn forward(&mut self, x: Array4<f32>) {
        
        let kernel_size = self.kernel_size;
        let x_shape = x.shape();

        
        let padding = self.padding;
        let stride = self.stride;
        let out_channel = self.out_channel;

        let padded = ConvLayer::add_padding(&x, padding);
        let padded_size = padded.shape();

        let sample_size =   padded_size[0];
        let in_channles =   padded_size[1];
        let image_height =  padded_size[2];
        let image_width =   padded_size[3];

        let output_height = (image_height - kernel_size) / stride + 1;
        let output_width = (image_width - kernel_size) / stride + 1;

        let mut output_matrix = Array4::<f32>::zeros((sample_size, out_channel, image_height, image_width));

        for sample in 0..sample_size {
            for out_chn in 0..out_channel {
                for oh in 0..output_height {
                    for ow in 0..output_width {
                        let mut result = 0.0;

                        for in_chn in 0..in_channles {
                            for i in 0..kernel_size {
                                for j in 0..kernel_size {
                                    let row = oh * stride + i;
                                    let col = ow * stride + j;
                                    result += padded[(sample, in_chn, row, col)] * self.kernel[[out_chn, i, j]];
                                }
                            }
                        }

                        output_matrix[[sample, out_chn, oh, ow]] = result;
                    }
                }
            }
        }

        self.forwarded = Some(output_matrix);

    }

    pub fn get_forwarded_output(&self) -> Array4<f32> {
        match &self.forwarded {
            Some(value) => value.clone(),
            None => panic!("No forwarded input give in convolutional layer"),
        }
    }
}
