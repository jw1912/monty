use crate::{Vector, Layer, Matrix, ReLU};

#[repr(C)]
#[derive(Clone, Copy)]
pub struct SubNet<const FEATS: usize> {
    ft: [Vector<16>; FEATS],
    l2: Layer<ReLU, 16, 16>,
}

impl<const FEATS: usize> std::ops::AddAssign<&SubNet<FEATS>>
    for SubNet<FEATS>
{
    fn add_assign(&mut self, rhs: &SubNet<FEATS>) {
        for (u, v) in self.ft.iter_mut().zip(rhs.ft.iter()) {
            *u += *v;
        }

        self.l2 += rhs.l2;
    }
}

impl<const FEATS: usize> SubNet<FEATS> {
    pub fn out(&self, feats: &[usize]) -> Vector<16> {
        self.l2.out(self.ft(feats))
    }

    pub fn out_with_layers(&self, feats: &[usize]) -> (Vector<16>, Vector<16>) {
        let ft = self.ft(feats);
        let l2 = self.l2.out(ft);
        (ft, l2)
    }

    pub fn backprop(
        &self,
        feats: &[usize],
        factor: f32,
        grad: &mut Self,
        other: Vector<16>,
        ft: Vector<16>,
        l2: Vector<16>,
    ) {
        let cumulated = factor * other;
        let mut cumulated = self.l2.backprop(&mut grad.l2, cumulated, ft, l2);

        cumulated = cumulated * ft.derivative::<ReLU>();
        for &feat in feats.iter() {
            grad.ft[feat] += cumulated;
        }
    }

    fn ft(&self, feats: &[usize]) -> Vector<16> {
        let mut res = Vector::zeroed();

        for &feat in feats {
            res += self.ft[feat];
        }

        res.activate::<ReLU>()
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

    pub const fn zeroed() -> Self {
        Self {
            ft: [Vector::zeroed(); FEATS],
            l2: Layer::from_raw(Matrix::from_raw([Vector::zeroed(); 16]), Vector::zeroed()),
        }
    }

    pub fn from_fn<F: FnMut() -> f32>(mut f: F) -> Self {
        let mut v = [Vector::zeroed(); 16];
        for r in v.iter_mut() {
            *r = Vector::from_fn(|_| f());
        }
        let m = Matrix::from_raw(v);

        let mut res = Self {
            ft: [Vector::zeroed(); FEATS],
            l2: Layer::from_raw(m, Vector::from_fn(|_| f())),
        };

        for v in res.ft.iter_mut() {
            *v = Vector::from_fn(|_| f());
        }

        res
    }
}
