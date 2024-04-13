#[derive(Clone, Copy, Debug)]
pub struct HashEntry {
    pub hash: u64,
    pub visits: i32,
    pub wins: f32,
    pub ptr: i32,
}

impl Default for HashEntry {
    fn default() -> Self {
        Self {
            hash: 0,
            visits: 0,
            wins: 0.0,
            ptr: 0,
        }
    }
}

pub struct HashTable {
    table: Vec<HashEntry>,
}

impl HashTable {
    pub fn new(size: usize) -> Self {
        Self { table: vec![HashEntry::default(); size] }
    }

    pub fn clear(&mut self) {
        for entry in &mut self.table {
            *entry = HashEntry::default();
        }
    }

    pub fn fetch(&self, hash: u64) -> &HashEntry {
        let idx = hash % (self.table.len() as u64);
        &self.table[idx as usize]
    }

    pub fn get(&self, hash: u64) -> Option<HashEntry> {
        let entry = self.fetch(hash);

        if entry.hash == hash {
            Some(*entry)
        } else {
            None
        }
    }

    pub fn push(&mut self, hash: u64, visits: i32, wins: f32, ptr: i32) {
        let idx = hash % (self.table.len() as u64);
        self.table[idx as usize] = HashEntry {
            hash,
            visits,
            wins,
            ptr,
        };
    }
}