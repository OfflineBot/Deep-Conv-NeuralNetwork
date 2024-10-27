
use ndarray::{Array1, Array2};

pub struct FullyConnected {
    pub(crate) weight: Array2<f32>,
    pub(crate) bias: Array1<f32>,
    pub(crate) forward: Option<Array2<f32>>,
    pub(crate) activated: Option<Array2<f32>>,
    pub(crate) activation: Option<Activation>,
    pub(crate) gradient: Option<Array2<f32>>,
}

pub fn relu(x: Array2<f32>, derivative: bool) -> Array2<f32> {
    match derivative {
        true => x.mapv(|f| if f > 0.0 { return 1.0 } else { 0.0 }),
        false => x.mapv(|f| if f > 0.0 { return f } else { 0.0 }),
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Activation {
    ReLU,
    Sigmoid
}

