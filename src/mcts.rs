use crate::position::{Position, MoveList, GameState, Move};

use std::time::Instant;

struct Node {
    visits: i32,
    wins: f64,
    left: usize,
    state: GameState,
    moves: MoveList,
}

impl Node {
    fn new(pos: &Position, root_stm: usize) -> Self {
        let moves = pos.gen();
        let state = pos.game_state(&moves, root_stm);

        Self {
            visits: 0,
            wins: 0.0,
            left: moves.len(),
            state,
            moves,
        }
    }

    fn is_terminal(&self) -> bool {
        self.state != GameState::Ongoing
    }

    fn num_children(&self) -> usize {
        self.moves.len()
    }
}

pub struct Searcher {
    startpos: Position,
    pos: Position,
    tree: Vec<Node>,
    node_limit: usize,
    selection: Vec<i32>,
    random: u64,
}

impl Searcher {
    pub fn new(pos: Position, node_limit: usize) -> Self {
        Self {
            startpos: pos,
            pos,
            tree: Vec::new(),
            node_limit,
            selection: Vec::new(),
            random: 21_976_391,
        }
    }

    fn selected(&self) -> i32 {
        *self.selection.last().unwrap()
    }

    fn random(&mut self) -> u64 {
        self.random ^= self.random << 13;
        self.random ^= self.random >> 7;
        self.random ^= self.random << 17;
        self.random
    }

    fn select_leaf(&mut self) {
        self.pos = self.startpos;
        self.selection.clear();
        self.selection.push(0);

        let mut node_ptr = 0;

        loop {
            let random = self.random() as usize;
            let node = &self.tree[node_ptr as usize];

            if node.is_terminal() {
                break;
            }

            let random_idx = random % node.num_children();
            let mov = node.moves[random_idx];
            let next = mov.ptr;

            if next == -1 {
                break;
            }

            self.pos.make(mov);
            self.selection.push(next);
            node_ptr = next;
        }
    }

    fn expand_node(&mut self) {
        let random = self.random() as usize;
        let node_ptr = self.selected();
        let node = &mut self.tree[node_ptr as usize];

        assert!(node.left > 0);

        let random_idx = random % node.left;
        node.left -= 1;

        if node.left > 0 {
            node.moves.swap(random_idx, node.left);
        }

        self.pos.make(node.moves[node.left]);

        let new_node = Node::new(&self.pos, self.startpos.stm());
        self.tree.push(new_node);

        let new_ptr = self.tree.len() as i32 - 1;
        let node = &mut self.tree[node_ptr as usize];
        let to_explore = &mut node.moves[node.left];
        to_explore.ptr = new_ptr;

        self.selection.push(to_explore.ptr);
    }

    fn simulate(&self) -> f64 {
        let node_ptr = self.selected();

        let node = &self.tree[node_ptr as usize];

        match node.state {
            GameState::Won => 1.0,
            GameState::Lost => 0.0,
            GameState::Draw => 0.5,
            GameState::Ongoing => {
                let stm_eval = self.pos.eval();
                if self.startpos.stm() == self.pos.stm() {
                    stm_eval
                } else {
                    1.0 - stm_eval
                }
            },
        }
    }

    fn backprop(&mut self, result: f64) {
        while let Some(node_ptr) = self.selection.pop() {
            let node = &mut self.tree[node_ptr as usize];
            node.visits += 1;
            node.wins += result;
        }

        self.tree[0].visits += 1;
    }

    fn get_bestmove(&self) -> (Move, f64) {
        let root_node = &self.tree[0];

        let mut best_move = root_node.moves[0];
        let mut best_score = -1.0;

        for mov in root_node.moves.iter() {
            let node = &self.tree[mov.ptr as usize];
            let score = node.wins / f64::from(node.visits);

            println!("info move {} score wdl {:.2}", mov.to_uci(), score * 100.0);

            if score > best_score {
                best_score = score;
                best_move = *mov;
            }
        }

        (best_move, best_score)
    }

    pub fn search(&mut self) -> (Move, f64) {
        let timer = Instant::now();
        self.tree.clear();

        let root_node = Node::new(&self.startpos, self.startpos.stm());
        self.tree.push(root_node);

        let mut nodes = 1;

        while nodes <= self.node_limit {
            self.select_leaf();

            if !self.tree[self.selected() as usize].is_terminal() {
                self.expand_node();
            }

            let result = self.simulate();

            self.backprop(result);

            if nodes % 100_000 == 0 {
                let (bm, score) = self.get_bestmove();
                let elapsed = timer.elapsed().as_secs_f32();
                let nps = nodes as f32 / elapsed;
                println!(
                    "info score cp {:.2} nodes {nodes} nps {nps:.0} pv {}",
                    -400.0 * (1.0 / score - 1.0).ln(),
                    bm.to_uci()
                );
            }

            nodes += 1;
        }

        self.get_bestmove()
    }
}