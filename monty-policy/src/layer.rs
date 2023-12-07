use std::marker::PhantomData;

use crate::{activation::Activation, Matrix, Vector};

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Layer<T: Activation, const M: usize, const N: usize> {
    weights: Matrix<N, M>,
    bias: Vector<N>,
    phantom: PhantomData<T>,
}

impl<T: Activation, const M: usize, const N: usize> Layer<T, M, N> {
    pub fn out(&self, inp: Vector<M>) -> Vector<N> {
        (self.weights * inp + self.bias).activate::<T>()
    }

    pub fn backprop(
        &self,
        factor: f32,
        grad: &mut Self,
        other: Vector<N>,
        inp: Vector<M>,
        out: Vector<N>,
    ) {
        let adj = factor * other * out.derivative::<T>();

        for (i, row) in grad.weights.iter_mut().enumerate() {
            *row += adj[i] * inp;
        }
    }
}
