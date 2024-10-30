
#![allow(unused)]

use ndarray::array;

mod network;

fn main() {
    let mut net = network::Network::new();

    let input_data = array![
        [
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ]
        ],
    ];

    net.add_conv(1, 2, 3, 2, 1);
    net.add_conv(2, 2, 3, 0, 1);
    net.add_conv(2, 2, 3, 0, 1);
    net.add_conv(2, 2, 3, 0, 1);
    net.add_conv(2, 2, 3, 0, 1);
    net.add_conv(2, 2, 3, 0, 1);
    net.add_conv(2, 2, 3, 0, 1);
    net.randomize(-0.5, 0.5);
    let x = net.forward_conv(input_data);
    println!("output: {:#?}", x);
}
