use std::marker::PhantomData;

use crate::{activation::Activation, Matrix, Vector};

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Layer<T: Activation, const M: usize, const N: usize> {
    weights: Matrix<N, M>,
    bias: Vector<N>,
    phantom: PhantomData<T>,
}

impl<T: Activation, const M: usize, const N: usize> std::ops::AddAssign<Layer<T, M, N>> for Layer<T, M, N> {
    fn add_assign(&mut self, rhs: Layer<T, M, N>) {
        self.weights += rhs.weights;
        self.bias += rhs.bias;
    }
}

impl<T: Activation, const M: usize, const N: usize> Layer<T, M, N> {
    pub const fn from_raw(weights: Matrix<N, M>, bias: Vector<N>) -> Self {
        Self { weights, bias, phantom: PhantomData }
    }

    pub fn out(&self, inp: Vector<M>) -> Vector<N> {
        (self.weights * inp + self.bias).activate::<T>()
    }

    pub fn transpose_mul(&self, out: Vector<N>) -> Vector<M> {
        self.weights.transpose_mul(out)
    }

    pub fn backprop(
        &self,
        grad: &mut Self,
        mut cumulated: Vector<N>,
        inp: Vector<M>,
        out: Vector<N>,
    ) -> Vector<M> {
        cumulated = cumulated * out.derivative::<T>();

        for (i, row) in grad.weights.iter_mut().enumerate() {
            *row += cumulated[i] * inp;
        }

        grad.bias += cumulated;
        self.transpose_mul(cumulated)
    }

    pub fn adam(
        &mut self,
        grad: &Self,
        momentum: &mut Self,
        velocity: &mut Self,
        adj: f32,
        lr: f32,
    ) {
        const B1: f32 = 0.9;
        const B2: f32 = 0.999;

        for i in 0..N {
            let g = adj * grad.weights[i];
            let m = &mut momentum.weights[i];
            let v = &mut velocity.weights[i];
            let p = &mut self.weights[i];

            *m = B1 * *m + (1. - B1) * g;
            *v = B2 * *v + (1. - B2) * g * g;
            *p -= lr * *m / (v.sqrt() + 0.000_000_01);
        }

        let g = adj * grad.bias;
        let m = &mut momentum.bias;
        let v = &mut velocity.bias;
        let p = &mut self.bias;

        *m = B1 * *m + (1. - B1) * g;
        *v = B2 * *v + (1. - B2) * g * g;
        *p -= lr * *m / (v.sqrt() + 0.000_000_01);
    }
}
