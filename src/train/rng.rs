pub struct Rand(u32);

impl Default for Rand {
    fn default() -> Self {
        Self(
            (std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("valid")
                .as_nanos()
                & 0xFFFF_FFFF) as u32,
        )
    }
}

impl Rand {
    pub fn new(seed: u32) -> Self {
        Self(seed)
    }

    pub fn rand_int(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        self.0
    }
}
