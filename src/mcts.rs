use crate::{
    moves::{Move, MoveList},
    params::TunableParams,
    position::{GameState, Position},
};

use std::{fmt::Write, time::Instant};

struct Node {
    visits: i32,
    wins: f64,
    left: usize,
    state: GameState,
    moves: MoveList,
}

impl Node {
    fn new(pos: &Position, stack: &[u64], params: &TunableParams) -> Self {
        let mut moves = pos.gen();
        let state = pos.game_state(&moves, stack);
        moves.set_policies(pos, params);

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
}

pub struct Searcher {
    startpos: Position,
    startstack: Vec<u64>,
    pos: Position,
    tree: Vec<Node>,
    stack: Vec<u64>,
    node_limit: usize,
    selection: Vec<i32>,
    params: TunableParams,
}

impl Searcher {
    pub fn new(pos: Position, stack: Vec<u64>, node_limit: usize, params: TunableParams) -> Self {
        Self {
            startpos: pos,
            startstack: stack.clone(),
            pos,
            tree: Vec::new(),
            stack,
            node_limit,
            selection: Vec::new(),
            params,
        }
    }

    fn make_move(&mut self, mov: Move) {
        self.stack.push(self.pos.hash());
        self.pos.make(mov);
    }

    fn selected(&self) -> i32 {
        *self.selection.last().unwrap()
    }

    fn pick_child(&self, node: &Node) -> usize {
        let expl = self.params.cpuct() * f64::from(node.visits).sqrt();

        let mut best_idx = 0;
        let mut best_uct = 0.0;

        for (idx, mov) in node.moves.iter().enumerate() {
            let uct = if mov.ptr() == -1 {
                self.params.fpu() + expl * mov.policy()
            } else {
                let child = &self.tree[mov.ptr() as usize];

                let q = child.wins / f64::from(child.visits);
                let u = expl * mov.policy() / f64::from(1 + child.visits);

                q + u
            };

            if uct > best_uct {
                best_uct = uct;
                best_idx = idx;
            }
        }

        best_idx
    }

    fn select_leaf(&mut self) {
        self.pos = self.startpos;
        self.stack = self.startstack.clone();
        self.selection.clear();
        self.selection.push(0);

        let mut node_ptr = 0;

        loop {
            let node = &self.tree[node_ptr as usize];

            if node.is_terminal() {
                break;
            }

            let mov_idx = self.pick_child(node);
            let mov = node.moves[mov_idx];
            let next = mov.ptr();

            if next == -1 {
                break;
            }

            self.make_move(mov);
            self.selection.push(next);
            node_ptr = next;
        }
    }

    fn expand_node(&mut self) {
        let node_ptr = self.selected();
        let node = &self.tree[node_ptr as usize];

        assert!(node.left > 0);

        let new_idx = self.pick_child(node);

        let node = &mut self.tree[node_ptr as usize];
        node.left -= 1;

        if node.left > 0 {
            node.moves.swap(new_idx, node.left);
        }

        let mov = node.moves[node.left];
        self.make_move(mov);

        let new_node = Node::new(&self.pos, &self.stack, &self.params);
        self.tree.push(new_node);

        let new_ptr = self.tree.len() as i32 - 1;
        let node = &mut self.tree[node_ptr as usize];
        let to_explore = &mut node.moves[node.left];
        to_explore.set_ptr(new_ptr);

        self.selection.push(to_explore.ptr());
    }

    fn simulate(&self) -> f64 {
        let node_ptr = self.selected();

        let node = &self.tree[node_ptr as usize];

        match node.state {
            GameState::Lost => -self.params.mate_bonus(),
            GameState::Draw => 0.5,
            GameState::Ongoing => self.pos.eval(&self.params),
        }
    }

    fn backprop(&mut self, mut result: f64) {
        while let Some(node_ptr) = self.selection.pop() {
            let node = &mut self.tree[node_ptr as usize];
            node.visits += 1;
            result = 1.0 - result;
            node.wins += result;
        }
    }

    fn get_bestmove<const REPORT: bool>(&self, root_node: &Node) -> (Move, f64) {
        let mut best_move = root_node.moves[0];
        let mut best_score = 0.0;

        for mov in root_node.moves.iter() {
            if mov.ptr() == -1 {
                continue;
            }

            let node = &self.tree[mov.ptr() as usize];
            let score = node.wins / f64::from(node.visits);

            if REPORT {
                println!(
                    "info move {} score wdl {:.2}% ({:.2} / {})",
                    mov.to_uci(),
                    score * 100.0,
                    node.wins,
                    node.visits,
                );
            }

            if score > best_score {
                best_score = score;
                best_move = *mov;
            }
        }

        (best_move, best_score)
    }

    fn get_pv(&self) -> (Vec<Move>, f64) {
        let mut node = &self.tree[0];

        let (mut mov, score) = self.get_bestmove::<false>(node);

        let mut pv = Vec::new();

        while mov.ptr() != -1 {
            pv.push(mov);
            node = &self.tree[mov.ptr() as usize];

            if node.moves.is_empty() {
                break;
            }

            mov = self.get_bestmove::<false>(node).0;
        }

        (pv, score)
    }

    pub fn search(
        &mut self,
        max_time: Option<u128>,
        max_depth: usize,
        report_moves: bool,
        uci_output: bool,
        total_nodes: &mut usize,
    ) -> (Move, f64) {
        let timer = Instant::now();
        self.tree.clear();

        let root_node = Node::new(&self.startpos, &[], &self.params);
        self.tree.push(root_node);

        let mut nodes = 1;
        let mut depth = 0;
        let mut seldepth = 0;
        let mut cumulative_depth = 0;

        while nodes <= self.node_limit {
            self.select_leaf();

            let this_depth = self.selection.len();
            cumulative_depth += this_depth;
            let avg_depth = cumulative_depth / nodes;
            seldepth = seldepth.max(this_depth);

            if !self.tree[self.selected() as usize].is_terminal() {
                self.expand_node();
            }

            let result = self.simulate();

            self.backprop(result);

            if let Some(time) = max_time {
                if nodes % 128 == 0 && timer.elapsed().as_millis() >= time {
                    break;
                }
            }

            if avg_depth > depth {
                depth = avg_depth;

                if uci_output {
                    let (pv_line, score) = self.get_pv();
                    let elapsed = timer.elapsed();
                    let nps = nodes as f32 / elapsed.as_secs_f32();
                    let pv = pv_line.iter().fold(String::new(), |mut pv_str, mov| {
                        write!(&mut pv_str, "{} ", mov.to_uci()).unwrap();
                        pv_str
                    });

                    println!(
                        "info depth {depth} \
                        seldepth {seldepth} \
                        score cp {:.0} \
                        time {} \
                        nodes {nodes} \
                        nps {nps:.0} \
                        pv {pv}",
                        -400.0 * (1.0 / score - 1.0).ln(),
                        elapsed.as_millis(),
                    );
                }
            }

            if depth >= max_depth {
                break;
            }

            nodes += 1;
        }

        *total_nodes += nodes;

        if report_moves {
            self.get_bestmove::<true>(&self.tree[0])
        } else {
            self.get_bestmove::<false>(&self.tree[0])
        }
    }
}
