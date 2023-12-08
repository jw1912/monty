use crate::{Vector, Layer, Matrix, ReLU, SparseLayer};

#[repr(C)]
#[derive(Clone, Copy)]
pub struct SubNet {
    ft: SparseLayer<ReLU, 768, 16>,
    l2: Layer<ReLU, 16, 16>,
}

impl std::ops::AddAssign<&SubNet> for SubNet {
    fn add_assign(&mut self, rhs: &SubNet) {
        self.ft += rhs.ft;
        self.l2 += rhs.l2;
    }
}

impl SubNet {
    pub fn out(&self, feats: &[usize]) -> Vector<16> {
        self.l2.out(self.ft.out(feats))
    }

    pub fn out_with_layers(&self, feats: &[usize]) -> (Vector<16>, Vector<16>) {
        let ft = self.ft.out(feats);
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
        let cumulated = self.l2.backprop(&mut grad.l2, cumulated, ft, l2);
        self.ft.backprop(&mut grad.ft, cumulated, feats, ft);
    }

    pub fn adam(
        &mut self,
        grad: &Self,
        momentum: &mut Self,
        velocity: &mut Self,
        adj: f32,
        lr: f32,
    ) {
        self.ft.adam(&grad.ft, &mut momentum.ft, &mut velocity.ft, adj, lr);

        self.l2.adam(&grad.l2, &mut momentum.l2, &mut velocity.l2, adj, lr);
    }

    pub const fn zeroed() -> Self {
        Self {
            ft: SparseLayer::from_raw(Matrix::zeroed(), Vector::zeroed()),
            l2: Layer::from_raw(Matrix::zeroed(), Vector::zeroed()),
        }
    }

    pub fn from_fn<F: FnMut() -> f32>(mut f: F) -> Self {
        let mut v = [Vector::zeroed(); 768];
        for r in v.iter_mut() {
            *r = Vector::from_fn(|_| f());
        }
        let m = Matrix::from_raw(v);

        let mut v2 = [Vector::zeroed(); 16];
        for r in v2.iter_mut() {
            *r = Vector::from_fn(|_| f());
        }
        let m2 = Matrix::from_raw(v2);

        Self {
            ft: SparseLayer::from_raw(m, Vector::from_fn(|_| f())),
            l2: Layer::from_raw(m2, Vector::from_fn(|_| f())),
        }
    }
}
