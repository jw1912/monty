use monty_core::{Flag, Move, Position, FeatureList};

pub static POLICY_NETWORK: PolicyNetwork =
    unsafe { std::mem::transmute(*include_bytes!("../../resources/policy.bin")) };

#[derive(Clone, Copy, Default)]
pub struct PolicyVal {
    left: f32,
    right: f32,
}

impl std::ops::Add<PolicyVal> for PolicyVal {
    type Output = PolicyVal;
    fn add(self, rhs: PolicyVal) -> Self::Output {
        PolicyVal { left: self.left + rhs.left, right: self.right + rhs.right }
    }
}

impl std::ops::Add<f32> for PolicyVal {
    type Output = PolicyVal;
    fn add(self, rhs: f32) -> Self::Output {
        PolicyVal { left: self.left + rhs, right: self.right + rhs }
    }
}

impl std::ops::AddAssign<PolicyVal> for PolicyVal {
    fn add_assign(&mut self, rhs: PolicyVal) {
        self.left += rhs.left;
        self.right += rhs.right;
    }
}

impl std::ops::Div<PolicyVal> for PolicyVal {
    type Output = PolicyVal;
    fn div(self, rhs: PolicyVal) -> Self::Output {
        PolicyVal { left: self.left / rhs.left, right: self.right / rhs.right }
    }
}

impl std::ops::Mul<PolicyVal> for PolicyVal {
    type Output = PolicyVal;
    fn mul(self, rhs: PolicyVal) -> Self::Output {
        PolicyVal { left: self.left * rhs.left, right: self.right * rhs.right }
    }
}

impl std::ops::Mul<PolicyVal> for f32 {
    type Output = PolicyVal;
    fn mul(self, rhs: PolicyVal) -> Self::Output {
        PolicyVal { left: self * rhs.left, right: self * rhs.right }
    }
}

impl std::ops::SubAssign<PolicyVal> for PolicyVal {
    fn sub_assign(&mut self, rhs: PolicyVal) {
        self.left -= rhs.left;
        self.right -= rhs.right;
    }
}

impl PolicyVal {
    pub fn out(&self) -> f32 {
        self.left.max(0.0) * self.right.max(0.0)
    }

    pub fn sqrt(self) -> Self {
        Self { left: self.left.sqrt(), right: self.right.sqrt() }
    }

    pub fn from_raw(left: f32, right: f32) -> Self {
        Self { left, right }
    }

    pub fn derivative(self) -> Self {
        Self {
            left: if self.left > 0.0 {1.0} else {0.0},
            right: if self.right > 0.0 {1.0} else {0.0},
        }
    }

    pub fn swap_relu(self) -> Self {
        Self { left: self.right.max(0.0), right: self.left.max(0.0) }
    }
}

pub struct NetworkDims;
impl NetworkDims {
    pub const INDICES: usize = 6 * 64;
    pub const FEATURES: usize = 769;
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PolicyNetwork {
    pub weights: [[PolicyVal; NetworkDims::FEATURES]; NetworkDims::INDICES],
}

impl std::ops::AddAssign<&PolicyNetwork> for PolicyNetwork {
    fn add_assign(&mut self, rhs: &PolicyNetwork) {
        for (i, j) in self.weights.iter_mut().zip(rhs.weights.iter()) {
            for (a, b) in i.iter_mut().zip(j.iter()) {
                *a += *b;
            }
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

        score.out()
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
