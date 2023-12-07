use std::marker::PhantomData;

use crate::{activation::Activation, Vector};

pub struct SubNet<T: Activation, const N: usize, const FEATS: usize> {
    ft: [Vector<N>; FEATS],
    phantom: PhantomData<T>,
}

impl<T: Activation, const N: usize, const FEATS: usize> SubNet<T, N, FEATS> {
    pub fn out<const M: usize>(&self, feats: &[usize]) -> Vector<N> {
        self.ft(feats).activate::<T>()
    }

    pub fn ft(&self, feats: &[usize]) -> Vector<N> {
        let mut res = Vector::<N>::zeroed();

        for &feat in feats {
            res += self.ft[feat];
        }

        res
    }

    pub fn backprop(
        &self,
        feats: &[usize],
        factor: f32,
        grad: &mut Self,
        other: Vector<N>,
        ft: &Vector<N>,
    ) {
        let adj = factor * other * ft.derivative::<T>();
        for &feat in feats.iter() {
            grad.ft[feat] += adj;
        }
    }
}
