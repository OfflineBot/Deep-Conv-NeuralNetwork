
use ndarray::Array2;

pub fn mse_loss(pred: &Array2<f32>, truth: &Array2<f32>) -> f32 {
    (pred-truth)
        .mapv(|f| f.powf(2.0))
        .flatten()
        .mean()
        .unwrap()
}
