use super::moves::Move;

use goober::{activation, layer, FeedForwardNetwork, Matrix, SparseVector, Vector};

#[repr(C)]
#[derive(Clone, Copy, FeedForwardNetwork)]
pub struct SubNet {
    ft: layer::SparseConnected<activation::ReLU, 2916, 8>,
}

impl SubNet {
    pub const fn zeroed() -> Self {
        Self {
            ft: layer::SparseConnected::zeroed(),
        }
    }

    pub fn from_fn<F: FnMut() -> f32>(mut f: F) -> Self {
        let matrix = Matrix::from_fn(|_, _| f());
        let vector = Vector::from_fn(|_| f());

        Self {
            ft: layer::SparseConnected::from_raw(matrix, vector),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PolicyNetwork {
    pub subnets: [SubNet; 99],
}

impl PolicyNetwork {
    pub fn get(&self, mov: &Move, feats: &SparseVector) -> f32 {
        let from_subnet = &self.subnets[mov.from().min(49)];
        let from_vec = from_subnet.out(feats);

        let to_subnet = &self.subnets[50 + mov.to().min(48)];
        let to_vec = to_subnet.out(feats);

        from_vec.dot(&to_vec)
    }
}
