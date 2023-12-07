pub trait Activation {
    fn activate(x: f32) -> f32;

    fn derivative(x: f32) -> f32;
}

pub struct ReLU;
impl Activation for ReLU {
    fn activate(x: f32) -> f32 {
        x.max(0.0)
    }

    fn derivative(x: f32) -> f32 {
        if x > 0.0 {1.0} else {0.0}
    }
}