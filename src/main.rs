
#![allow(unused)]

use ndarray::array;

mod network;

fn main() {
    let mut net = network::Network::new();
    let activation = network::Activation::ReLU;

    let input = array![
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.0, 0.0],
    ];

    let input_data = (&input - &input.mean_axis(ndarray::Axis(0)).unwrap() / &input.std_axis(ndarray::Axis(0), 0.0).mapv(|f| if f == 0.0 { return 0.001 } else { f }));

    let output_data = array![
        [1.0],
        [1.0],
        [0.0],
        [0.0],
    ];

    net.add_fc(2, 10, Some(activation.clone()));
    net.add_fc(10, 10, Some(activation.clone()));
    net.add_fc(10, 1, None);
    net.randomize(-0.5, 0.5);

    let iterations = 1000;
    let debug_print = iterations / 10;
    let learning_rate = 0.01;
    let loss_break = 0.0001;

    for i in 0..=iterations {
        let pred = net.forward_fc(input_data.clone());

        let loss = network::mse_loss(&pred, &output_data);
        if loss < loss_break {
            println!("Loss smaller than loss_break at iteration: {}", i);
            break;
        }
        if i % debug_print == 0 {
            println!("Loss: {} | Iteration: {}", loss, i);
        }
        net.backward(&output_data, &pred);
        net.update(learning_rate, &input_data);
    }

    let out = net.forward_fc(input_data.clone());
    println!("out: {:#?}", out);

}
