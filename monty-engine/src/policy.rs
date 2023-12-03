use monty_core::{Flag, Move, Position, FeatureList};

pub static POLICY_NETWORK: PolicyNetwork =
    //unsafe { std::mem::transmute(*include_bytes!("../../resources/policy.bin")) };
PolicyNetwork {
    weights: [[PolicyVal::from_raw([0.0; NetworkDims::NEURONS]); NetworkDims::FEATURES]; NetworkDims::INDICES],
    outputs: [0.0; NetworkDims::NEURONS],
};

pub struct NetworkDims;
impl NetworkDims {
    pub const INDICES: usize = 6 * 64;
    pub const FEATURES: usize = 769;
    pub const NEURONS: usize = 8;
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PolicyNetwork {
    pub weights: [[PolicyVal; NetworkDims::FEATURES]; NetworkDims::INDICES],
    pub outputs: [f32; NetworkDims::NEURONS],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct PolicyVal {
    inner: [f32; NetworkDims::NEURONS],
}

impl std::ops::Add<PolicyVal> for PolicyVal {
    type Output = PolicyVal;
    fn add(mut self, rhs: PolicyVal) -> Self::Output {
        for (i, j) in self.inner.iter_mut().zip(rhs.inner.iter()) {
            *i += *j;
        }

        self
    }
}

impl std::ops::Add<f32> for PolicyVal {
    type Output = PolicyVal;
    fn add(mut self, rhs: f32) -> Self::Output {
        for i in self.inner.iter_mut() {
            *i += rhs;
        }

        self
    }
}

impl std::ops::AddAssign<PolicyVal> for PolicyVal {
    fn add_assign(&mut self, rhs: PolicyVal) {
        for (i, j) in self.inner.iter_mut().zip(rhs.inner.iter()) {
            *i += *j;
        }
    }
}

impl std::ops::Div<PolicyVal> for PolicyVal {
    type Output = PolicyVal;
    fn div(mut self, rhs: PolicyVal) -> Self::Output {
        for (i, j) in self.inner.iter_mut().zip(rhs.inner.iter()) {
            *i /= *j;
        }

        self
    }
}

impl std::ops::Mul<PolicyVal> for PolicyVal {
    type Output = PolicyVal;
    fn mul(mut self, rhs: PolicyVal) -> Self::Output {
        for (i, j) in self.inner.iter_mut().zip(rhs.inner.iter()) {
            *i *= *j;
        }

        self
    }
}

impl std::ops::Mul<PolicyVal> for f32 {
    type Output = PolicyVal;
    fn mul(self, mut rhs: PolicyVal) -> Self::Output {
        for i in rhs.inner.iter_mut() {
            *i *= self;
        }

        rhs
    }
}

impl std::ops::SubAssign<PolicyVal> for PolicyVal {
    fn sub_assign(&mut self, rhs: PolicyVal) {
        for (i, j) in self.inner.iter_mut().zip(rhs.inner.iter()) {
            *i -= *j;
        }
    }
}

impl PolicyVal {
    pub fn out(&self, policy: &PolicyNetwork) -> f32 {
        let mut score = 0.0;
        for (i, j) in self.inner.iter().zip(policy.outputs.iter()) {
            score += i.max(0.0) * j;
        }

        score
    }

    pub fn sqrt(mut self) -> Self {
        for i in self.inner.iter_mut() {
            *i = i.sqrt();
        }

        self
    }

    pub const fn from_raw(inner: [f32; NetworkDims::NEURONS]) -> Self {
        Self { inner }
    }

    pub fn derivative(mut self) -> Self {
        for i in self.inner.iter_mut() {
            *i =if *i > 0.0 {1.0} else {0.0};
        }

        self
    }
}

impl std::ops::AddAssign<&PolicyNetwork> for PolicyNetwork {
    fn add_assign(&mut self, rhs: &PolicyNetwork) {
        for (i, j) in self.weights.iter_mut().zip(rhs.weights.iter()) {
            for (a, b) in i.iter_mut().zip(j.iter()) {
                *a += *b;
            }
        }

        for (i, j) in self.outputs.iter_mut().zip(rhs.outputs.iter()) {
            *i += *j;
        }
    }
}

impl PolicyNetwork {
    pub fn boxed_and_zeroed() -> Box<Self> {
        unsafe {
            let layout = std::alloc::Layout::new::<Self>();
            let ptr = std::alloc::alloc_zeroed(layout);
            if ptr.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            Box::from_raw(ptr.cast())
        }
    }

    pub fn write_to_bin(&self, path: &str) {
        use std::io::Write;
        const SIZEOF: usize = std::mem::size_of::<PolicyNetwork>();

        let mut file = std::fs::File::create(path).unwrap();

        unsafe {
            let ptr: *const Self = self;
            let slice_ptr: *const u8 = std::mem::transmute(ptr);
            let slice = std::slice::from_raw_parts(slice_ptr, SIZEOF);
            file.write_all(slice).unwrap();
        }
    }

    fn get_neuron(&self, idx: usize, feats: &FeatureList) -> f32 {
        let wref = &self.weights[idx];
        let mut score = PolicyVal::default();

        for &feat in feats.iter() {
            score += wref[feat];
        }

        score.out(self)
    }

    pub fn hce(mov: &Move, pos: &Position) -> f32 {
        let mut score = 0.0;

        if pos.see(mov, -108) {
            score += 2.0;
        }

        if [Flag::QPR, Flag::QPC].contains(&mov.flag()) {
            score += 2.0;
        }

        if mov.is_capture() {
            score += 2.0;

            let diff = pos.get_pc(1 << mov.to()) as i32 - i32::from(mov.moved());
            score += 0.2 * diff as f32;
        }

        score
    }

    pub fn get(mov: &Move, pos: &Position, policy: &PolicyNetwork, feats: &FeatureList) -> f32 {
        let idx = mov.index(pos.flip_val());
        let sq_policy = policy.get_neuron(idx, feats);

        let hce_policy = PolicyNetwork::hce(mov, pos);

        sq_policy + hce_policy
    }
}
