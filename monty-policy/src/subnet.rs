use std::marker::PhantomData;

use crate::{activation::Activation, Vector, Layer, Matrix};

#[repr(C)]
#[derive(Clone, Copy)]
pub struct SubNet<T: Activation, const N: usize, const FEATS: usize> {
    ft: [Vector<N>; FEATS],
    l2: Layer<T, N, N>,
    phantom: PhantomData<T>,
}

impl<T: Activation, const N: usize, const FEATS: usize> std::ops::AddAssign<&SubNet<T, N, FEATS>>
    for SubNet<T, N, FEATS>
{
    fn add_assign(&mut self, rhs: &SubNet<T, N, FEATS>) {
        for (u, v) in self.ft.iter_mut().zip(rhs.ft.iter()) {
            *u += *v;
        }

        self.l2 += rhs.l2;
    }
}

impl<T: Activation, const N: usize, const FEATS: usize> SubNet<T, N, FEATS> {
    pub const fn zeroed() -> Self {
        Self {
            ft: [Vector::zeroed(); FEATS],
            l2: Layer::from_raw(Matrix::from_raw([Vector::zeroed(); N]), Vector::zeroed()),
            phantom: PhantomData,
        }
    }

    pub fn from_fn<F: FnMut() -> f32>(mut f: F) -> Self {

        let mut v = [Vector::zeroed(); N];
        for r in v.iter_mut() {
            *r = Vector::from_fn(|_| f());
        }
        let m = Matrix::from_raw(v);

        let mut res = Self {
            ft: [Vector::zeroed(); FEATS],
            l2: Layer::from_raw(m, Vector::from_fn(|_| f())),
            phantom: PhantomData,
        };

        for v in res.ft.iter_mut() {
            *v = Vector::from_fn(|_| f());
        }

        res
    }

    pub fn out(&self, feats: &[usize]) -> Vector<N> {
        self.l2.out(self.ft(feats))
    }

    pub fn out_with_layers(&self, feats: &[usize]) -> (Vector<N>, Vector<N>) {
        let ft = self.ft(feats);
        let l2 = self.l2.out(ft);
        (ft, l2)
    }

    fn ft(&self, feats: &[usize]) -> Vector<N> {
        let mut res = Vector::zeroed();

        for &feat in feats {
            res += self.ft[feat];
        }

        res.activate::<T>()
    }

    pub fn backprop(
        &self,
        feats: &[usize],
        factor: f32,
        grad: &mut Self,
        other: Vector<N>,
        ft: Vector<N>,
        l2: Vector<N>,
    ) {
        let cumulated = factor * other * l2.derivative::<T>();
        self.l2.backprop(&mut grad.l2, cumulated, ft);

        let cumulated = self.l2.transpose_mul(cumulated) * ft.derivative::<T>();
        for &feat in feats.iter() {
            grad.ft[feat] += cumulated;
        }
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

        for i in 0..FEATS {
            let g = adj * grad.ft[i];
            let m = &mut momentum.ft[i];
            let v = &mut velocity.ft[i];
            let p = &mut self.ft[i];

            *m = B1 * *m + (1. - B1) * g;
            *v = B2 * *v + (1. - B2) * g * g;
            *p -= lr * *m / (v.sqrt() + 0.000_000_01);
        }

        self.l2.adam(&grad.l2, &mut momentum.l2, &mut velocity.l2, adj, lr);
    }
}
