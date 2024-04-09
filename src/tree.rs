mod edge;
mod node;
mod table;

pub use edge::Edge;
pub use node::{Mark, Node};
use table::HashTable;

use std::time::Instant;

use crate::games::{GameRep, GameState};

pub struct Tree {
    tree: Vec<Node>,
    _table: HashTable,
    root: i32,
    empty: i32,
    used: usize,
    mark: Mark,
}

impl std::ops::Index<i32> for Tree {
    type Output = Node;

    fn index(&self, index: i32) -> &Self::Output {
        &self.tree[index as usize]
    }
}

impl std::ops::IndexMut<i32> for Tree {
    fn index_mut(&mut self, index: i32) -> &mut Self::Output {
        &mut self.tree[index as usize]
    }
}

impl Tree {
    pub fn new_mb(mb: usize) -> Self {
        let bytes = mb * 1024 * 1024;

        let tree_bytes = 7 * bytes / 8;
        let tt_bytes = bytes / 8;

        let tree_cap = tree_bytes / std::mem::size_of::<Node>();
        let tt_cap = tt_bytes / std::mem::size_of::<i32>();

        Self::new(tree_cap, tt_cap)
    }

    fn new(tree_cap: usize, tt_cap: usize) -> Self {
        let mut tree = Self {
            tree: vec![Node::new(GameState::Ongoing); tree_cap],
            _table: HashTable::with_capacity(tt_cap),
            root: -1,
            empty: 0,
            used: 0,
            mark: Mark::Var1,
        };

        let end = tree.cap() as i32 - 1;

        for i in 0..end {
            tree[i].set_fwd_link(i + 1);
        }

        tree[end].set_fwd_link(-1);

        tree
    }

    pub fn push(&mut self, node: Node) -> i32 {
        let new = self.empty;

        assert_ne!(new, -1);

        self.used += 1;
        self.empty = self[self.empty].fwd_link();
        self[new] = node;

        let mark = self.mark;
        self[new].set_mark(mark);

        new
    }

    pub fn delete(&mut self, ptr: i32) {
        self[ptr] = Node::new(GameState::Ongoing);

        let empty = self.empty;
        self[ptr].set_fwd_link(empty);

        self.empty = ptr;
        self.used -= 1;
        assert!(self.used < self.cap());
    }

    pub fn root_node(&self) -> i32 {
        self.root
    }

    pub fn cap(&self) -> usize {
        self.tree.len()
    }

    pub fn len(&self) -> usize {
        self.used
    }

    pub fn remaining(&self) -> usize {
        self.cap() - self.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn clear(&mut self) {
        if self.used == 0 {
            return;
        }

        let root = self.root_node();
        self.delete_subtree(root, self[root].mark());
        assert_eq!(self.used, 0);
        assert_eq!(self.empty, root);
        self.root = -1;
        self.mark = Mark::Empty;
    }

    fn delete_subtree(&mut self, ptr: i32, bad_mark: Mark) {
        if self[ptr].mark() == bad_mark {
            for i in 0..self[ptr].actions().len() {
                let ptr = self[ptr].actions()[i].ptr();
                self.delete_subtree(ptr, bad_mark);
            }

            self.delete(ptr);
        }
    }

    pub fn make_root_node(&mut self, node: i32) {
        self.root = node;
        self.mark = self[node].mark();
        self[node].set_state(GameState::Ongoing);
    }

    pub fn try_use_subtree<T: GameRep>(&mut self, root: &T, prev_board: &Option<T>) {
        let t = Instant::now();

        if self.is_empty() {
            return;
        }

        println!("info string attempting to reuse tree");

        if let Some(board) = prev_board {
            println!("info string searching for subtree");

            let root = self.recurse_find(self.root, board, root, 2);

            if root == -1 || !self[root].has_children() {
                self.clear();
            } else if root != self.root_node() {
                let old_root = self.root_node();
                self.mark_subtree(root);
                self.make_root_node(root);
                self.delete_subtree(old_root, self[old_root].mark());

                println!("info string found subtree of size {} nodes", self.len());
            } else {
                println!(
                    "info string using current tree of size {} nodes",
                    self.len()
                );
            }
        } else {
            self.clear();
        }

        println!(
            "info string tree processing took {} microseconds",
            t.elapsed().as_micros()
        );
    }

    fn recurse_find<T: GameRep>(&self, start: i32, this_board: &T, board: &T, depth: u8) -> i32 {
        if this_board.is_same(board) {
            return start;
        }

        if start == -1 || depth == 0 {
            return -1;
        }

        let node = &self.tree[start as usize];

        for action in node.actions() {
            let child_idx = action.ptr();
            let mut child_board = this_board.clone();

            child_board.make_move(T::Move::from(action.mov()));

            let found = self.recurse_find(child_idx, &child_board, board, depth - 1);

            if found != -1 {
                return found;
            }
        }

        -1
    }

    fn mark_subtree(&mut self, ptr: i32) {
        let mark = self[ptr].mark();
        self[ptr].set_mark(mark.flip());

        for i in 0..self[ptr].actions().len() {
            let ptr = self[ptr].actions()[i].ptr();
            self.mark_subtree(ptr);
        }
    }

    pub fn get_best_child_by_key<F: FnMut(&Edge) -> f32>(&self, ptr: i32, mut key: F) -> &Edge {
        let mut best_child = &self[ptr].actions()[0];
        let mut best_score = f32::NEG_INFINITY;

        for action in self[ptr].actions() {
            let score = key(action);

            if score > best_score {
                best_score = score;
                best_child = action;
            }
        }

        best_child
    }

    pub fn get_best_child(&self, ptr: i32) -> &Edge {
        self.get_best_child_by_key(ptr, |child| self[child.ptr()].q())
    }

    pub fn display<T: GameRep>(&self, idx: i32, depth: usize) {
        let mut bars = vec![true; depth + 1];
        self.display_recurse::<T>(idx, depth + 1, 0, &mut bars, 0, 1.0);
    }

    fn display_recurse<T: GameRep>(
        &self,
        idx: i32,
        depth: usize,
        ply: usize,
        bars: &mut [bool],
        mov: u16,
        policy: f32,
    ) {
        let node = &self[idx];

        if depth == 0 || node.visits() == 0 {
            return;
        }

        let mov = if ply > 0 {
            for &bar in bars.iter().take(ply - 1) {
                if bar {
                    print!("\u{2502}   ");
                } else {
                    print!("    ");
                }
            }

            if bars[ply - 1] {
                print!("\u{251C}\u{2500}> ");
            } else {
                print!("\u{2514}\u{2500}> ");
            }

            T::Move::from(mov).to_string()
        } else {
            "root".to_string()
        };

        let mut q = node.q();
        if ply % 2 == 0 {
            q = 1.0 - q;
        }

        println!(
            "{mov} Q({:.2}%) N({}) P({:.2}%) S({})",
            q * 100.0,
            node.visits(),
            policy * 100.0,
            node.state(),
        );

        let mut active = Vec::new();
        for action in node.actions() {
            let child = &self[action.ptr()];
            if child.visits() > 0 {
                active.push((action.ptr(), action.mov(), action.policy()));
            }
        }

        let end = active.len() - 1;

        for (i, &(ptr, mov, policy)) in active.iter().enumerate() {
            if i == end {
                bars[ply] = false;
            }
            self.display_recurse::<T>(ptr, depth - 1, ply + 1, bars, mov, policy);
            bars[ply] = true;
        }
    }
}
