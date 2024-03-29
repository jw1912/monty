const HIDDEN: usize = 8;
const SCALE: i32 = 400;
const QA: i32 = 255;
const QB: i32 = 64;
const QAB: i32 = QA * QB;

#[inline]
fn activate(x: i16) -> i32 {
    i32::from(x).clamp(0, QA).pow(2)
}

#[repr(C, align(64))]
pub struct ValueNetwork {
    feature_weights: [Accumulator; 98],
    feature_bias: Accumulator,
    output_weights: Accumulator,
    output_bias: i16,
}

pub static NNUE: ValueNetwork =
    unsafe { std::mem::transmute(*include_bytes!("../../resources/ataxx-value.bin")) };

impl ValueNetwork {
    pub fn out(boys: &Accumulator) -> i32 {
        let mut sum = 0;

        for (&x, &w) in boys.vals.iter().zip(&NNUE.output_weights.vals) {
            sum += activate(x) * i32::from(w);
        }

        (sum / QA + i32::from(NNUE.output_bias)) * SCALE / QAB
    }
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct Accumulator {
    vals: [i16; HIDDEN],
}

impl Accumulator {
    pub fn add(&mut self, idx: usize) {
        assert!(idx < 768);
        for (i, d) in self.vals.iter_mut().zip(&NNUE.feature_weights[idx].vals) {
            *i += *d
        }
    }
}

impl Default for Accumulator {
    fn default() -> Self {
        NNUE.feature_bias
    }
}