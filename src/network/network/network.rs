
use ndarray::{Array1, Array2};

use crate::network::{fully_connected::fully_connected::FullyConnected, ConvLayer};

pub struct Network {
    pub(crate) convolutions: Vec<ConvLayer>,
    pub(crate) fully_connected: Vec<FullyConnected>,
}

