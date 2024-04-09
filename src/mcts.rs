pub use crate::params::MctsParams;
pub use crate::tree::{Edge, Mark, Node, Tree};

use crate::games::{GameRep, GameState};

use std::time::Instant;

#[derive(Clone, Copy)]
pub struct Limits {
    pub max_time: Option<u128>,
    pub max_depth: usize,
    pub max_nodes: usize,
}

pub struct Searcher<T: GameRep> {
    root_position: T,
    tree: Tree,
    selection: Vec<i32>,
    params: MctsParams,
}

impl<T: GameRep> Searcher<T> {
    pub fn new(root_position: T, tree: Tree, params: MctsParams) -> Self {
        Self {
            root_position,
            tree,
            selection: Vec::new(),
            params,
        }
    }

    /// the main MCTS search function
    pub fn search(
        &mut self,
        limits: Limits,
        uci_output: bool,
        total_nodes: &mut usize,
        prev_board: &Option<T>,
    ) -> (T::Move, f32) {
        let timer = Instant::now();

        // attempt to reuse the current tree stored in memory
        self.tree.try_use_subtree(&self.root_position, prev_board);

        // we failed to reuse a tree, push the root node to
        // the tree and expand it
        if self.tree.is_empty() {
            let node = self.tree.push(Node::new(GameState::Ongoing));
            self.tree.make_root_node(node);
            self.tree[node].expand::<T, true>(&self.root_position, &self.params);
        } else {
            let node = self.tree.root_node();
            self.tree[node].relabel_policy(&self.root_position, &self.params);
        }

        let mut nodes = 0;
        let mut depth = 0;
        let mut cumulative_depth = 0;

        // search until a further iteration may overflow the tree
        while self.tree.remaining() > 0 {
            nodes += 1;

            // start from the root
            let mut pos = self.root_position.clone();

            // step 1: select a leaf node to expand,
            // where a leaf node is defined as a node
            // which has children that have not yet been
            // expanded
            if let Some(action) = self.select_leaf(&mut pos) {
                // step 2: expand to the selected node
                // (mostly delayed to second visit)
                self.expand(action, &mut pos);
            }

            // update depth statistics
            let this_depth = self.selection.len() - 1;
            cumulative_depth += this_depth;
            let avg_depth = cumulative_depth / nodes;

            // step 3: simulate the game outcome
            let result = self.simulate(&pos);

            // step 4: backpropogate the result to the root
            self.backprop(result);

            if self.tree[self.tree.root_node()].is_terminal() {
                break;
            }

            // check if hit node limit
            if nodes >= limits.max_nodes {
                break;
            }

            // check for timeup
            if let Some(time) = limits.max_time {
                if nodes % 128 == 0 && timer.elapsed().as_millis() >= time {
                    break;
                }
            }

            // we define "depth" in the UCI sense as the average
            // depth of selection
            if avg_depth > depth {
                depth = avg_depth;

                if depth >= limits.max_depth {
                    break;
                }

                if uci_output {
                    self.search_report(depth, &timer, nodes);
                }
            }
        }

        *total_nodes += nodes;

        if uci_output {
            self.search_report(depth, &timer, nodes);
        }

        let idx = self.tree.get_best_child(self.tree.root_node());
        let best_child = &self.tree[self.tree.root_node()].actions()[idx];
        (T::Move::from(best_child.mov()), self.tree[best_child.ptr()].q())
    }

    fn select_leaf(&mut self, pos: &mut T) -> Option<usize> {
        // always start from the root
        let mut node_ptr = self.tree.root_node();

        self.selection.clear();
        self.selection.push(node_ptr);

        loop {
            let node = &self.tree[node_ptr];

            // if the node is terminal we can go no further,
            // we simply backpropogate the terminal score
            if node.is_terminal() {
                return None;
            }

            // this is "expanding on the second visit",
            // an important optimisation - not only does it
            // massively reduce memory usage, it also is a
            // large speedup (avoids many policy net calculations)
            if node.visits() == 1 && node.is_not_expanded() {
                self.tree[node_ptr].expand::<T, false>(pos, &self.params);
            }

            // pick the next child based on PUCT score
            let idx = self.pick_action(node_ptr);

            // proved a loss from the child nodes
            if idx == usize::MAX {
                return None;
            }

            let edge = &self.tree[node_ptr].actions()[idx];
            let mov = edge.mov();
            node_ptr = edge.ptr();

            if node_ptr == -1 {
                return Some(idx);
            }

            // descend down the tree
            pos.make_move(T::Move::from(mov));
            self.selection.push(node_ptr);
        }
    }

    fn expand(&mut self, action: usize, pos: &mut T) {
        // selected node
        let node_ptr = self.selected();
        let edge = &self.tree[node_ptr].actions()[action];

        let mov = T::Move::from(edge.mov());
        pos.make_move(mov);

        let state = pos.game_state();
        let ptr = self.tree.push(Node::new(state));

        self.tree[node_ptr].actions_mut()[action].set_ptr(ptr);
        self.selection.push(ptr);
    }

    fn simulate(&self, pos: &T) -> f32 {
        let node = &self.tree[self.selected()];

        // simulate the game outcome
        match node.state() {
            GameState::Ongoing => pos.get_value_wdl(),
            GameState::Draw => 0.5,
            GameState::Lost(_) => 0.0,
            GameState::Won(_) => 1.0,
        }
    }

    fn backprop(&mut self, mut result: f32) {
        let mut prev = GameState::Ongoing;
        while let Some(node_ptr) = self.selection.pop() {
            // flip result
            result = 1.0 - result;

            // for a `node` with given stm, `node.wins`
            // is stored from the nstm perspective, for
            // simplicity when it is used
            self.tree[node_ptr].update(1, result);

            if let GameState::Lost(n) = prev {
                self.tree[node_ptr].set_state(GameState::Won(n + 1));
            }

            prev = self.tree[node_ptr].state();
        }
    }

    fn pick_action(&mut self, ptr: i32) -> usize {
        if !self.tree[ptr].has_children() {
            panic!("trying to pick from no children!");
        }

        let is_root = ptr == self.tree.root_node();
        let cpuct = if is_root {
            self.params.root_cpuct()
        } else {
            self.params.cpuct()
        };
        let node = &self.tree[ptr];

        // exploration factor to apply
        let expl = cpuct * (node.visits().max(1) as f32).sqrt();

        // first play urgency - choose a Q value for
        // moves which have no been played yet
        let fpu = if node.visits() > 0 {
            1.0 - node.q()
        } else {
            0.5
        };

        let mut proven_loss = true;
        let mut win_len = 0;
        let mut best = 0;
        let mut max = f32::NEG_INFINITY;

        // return child with highest PUCT score
        for (i, action) in node.actions().iter().enumerate() {
            let puct = if action.ptr() == -1 {
                proven_loss = false;
                fpu + expl * action.policy()
            } else {
                let child = &self.tree[action.ptr()];

                if let GameState::Won(n) = child.state() {
                    win_len = n.max(win_len);
                } else {
                    proven_loss = false;
                }

                child.q() + expl * action.policy() / (1 + child.visits()) as f32
            };

            if puct > max {
                max = puct;
                best = i;
            }
        }

        if proven_loss {
            self.tree[ptr].set_state(GameState::Lost(win_len + 1));
            return usize::MAX;
        }

        best
    }

    fn search_report(&self, depth: usize, timer: &Instant, nodes: usize) {
        print!("info depth {depth} ");
        let (pv_line, score) = self.get_pv(depth);

        if score == 1.0 {
            print!("score mate {} ", (pv_line.len() + 1) / 2);
        } else if score == 0.0 {
            print!("score mate -{} ", pv_line.len() / 2);
        } else {
            let cp = -400.0 * (1.0 / score.clamp(0.0, 1.0) - 1.0).ln();
            print!("score cp {cp:.0} ");
        }

        let elapsed = timer.elapsed();
        let nps = nodes as f32 / elapsed.as_secs_f32();
        let ms = elapsed.as_millis();
        let hf = self.tree.len() * 1000 / self.tree.cap();

        print!("time {ms} nodes {nodes} nps {nps:.0} hashfull {hf} pv");

        for mov in pv_line {
            print!(" {}", self.root_position.conv_mov_to_str(mov));
        }

        println!();
    }

    fn get_pv(&self, mut depth: usize) -> (Vec<T::Move>, f32) {
        let key = |edge: &Edge| {
            if edge.ptr() == -1 {
                -10000.0
            } else {
                let child = &self.tree[edge.ptr()];
                match child.state() {
                    GameState::Draw => 0.5,
                    GameState::Ongoing => child.q(),
                    GameState::Lost(n) => 1.0 + f32::from(n),
                    GameState::Won(n) => f32::from(n) - 256.0,
                }
            }
        };

        let mate = self.tree[self.tree.root_node()].is_terminal();

        let idx = self.tree.get_best_child_by_key(self.tree.root_node(), key);
        let mut action = &self.tree[self.tree.root_node()].actions()[idx];

        if action.ptr() == -1 {
            println!("{:#?}", self.tree[self.tree.root_node()]);
        }

        let score = self.tree[action.ptr()].q();
        let mut pv = Vec::new();

        while (mate || depth > 0) && action.ptr() != -1 {
            pv.push(T::Move::from(action.mov()));
            let idx = self.tree.get_best_child_by_key(action.ptr(), key);

            if idx == usize::MAX {
                break;
            }

            action = &self.tree[action.ptr()].actions()[idx];
            depth -= 1;
        }

        (pv, score)
    }

    pub fn tree_and_board(self) -> (Tree, T) {
        (self.tree, self.root_position)
    }

    fn selected(&self) -> i32 {
        *self.selection.last().unwrap()
    }

    pub fn display_moves(&self) {
        for action in self.tree[self.tree.root_node()].actions() {
            let mov = self.root_position.conv_mov_to_str(action.mov().into());
            let q = self.tree[action.ptr()].q() * 100.0;
            println!("{mov} -> {q:.2}%");
        }
    }
}
