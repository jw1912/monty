use crate::{
    search::{mcts::Searcher, params::TunableParams, policy::PolicyNetwork},
    state::{position::{GameState, Position}, consts::Side},
    train::rng::Rand, uci::STARTPOS,
};

use std::time::{SystemTime, UNIX_EPOCH};

pub struct TrainingPosition {
    pub position: Position,
    pub moves: Vec<(usize, f64)>,
}

pub fn run_datagen(
    num_positions: usize,
    params: TunableParams,
    policy: &PolicyNetwork,
) -> Vec<TrainingPosition> {
    let mut thread = DatagenThread::new(params, policy);

    thread.run(num_positions);

    thread.positions
}

pub struct DatagenThread<'a> {
    id: u32,
    rng: Rand,
    params: TunableParams,
    policy: &'a PolicyNetwork,
    positions: Vec<TrainingPosition>,
}

impl<'a> DatagenThread<'a> {
    fn new(params: TunableParams, policy: &'a PolicyNetwork) -> Self {
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Guaranteed increasing.")
            .as_micros() as u32;

        let res = Self {
            id: seed,
            rng: Rand::new(seed),
            params,
            policy,
            positions: Vec::new(),
        };

        println!("thread id {} created", res.id);
        res
    }

    fn run(&mut self, num_positions: usize) {
        let position = Position::parse_fen(STARTPOS);

        while self.positions.len() < num_positions {
            self.run_game(position, self.params.clone(), self.policy);
        }
    }

    fn run_game(
        &mut self,
        position: Position,
        params: TunableParams,
        policy: &'a PolicyNetwork,
    ) {
        let mut engine = Searcher::new(position, Vec::new(), 1_000, params, policy);

        // play 8 or 9 random moves
        for _ in 0..(8 + (self.rng.rand_int() % 2)) {
            let moves = engine.startpos.gen();

            if moves.is_empty() {
                return;
            }

            let mov = moves[self.rng.rand_int() as usize % moves.len()];

            engine.startstack.push(engine.startpos.hash());
            engine.startpos.make(mov);
        }

        // play out game
        loop {
            let (bm, _) = engine.search(None, 128, false, false, &mut 0);

            let mut training_pos = TrainingPosition {
                position: engine.startpos,
                moves: Vec::new(),
            };

            for mov in engine.tree[0].moves.iter() {
                if mov.ptr() == -1 {
                    continue;
                }

                let child = &engine.tree[mov.ptr() as usize];
                let score = child.score();

                let flip = if engine.startpos.stm() == Side::BLACK {56} else {0};
                let idx = mov.index(flip);

                training_pos.moves.push((idx, score));
            }

            self.positions.push(training_pos);
            if self.positions.len() % 1024 == 0 {
                println!("thread {} count {}", self.id, self.positions.len());
            }

            engine.startstack.push(engine.startpos.hash());
            engine.startpos.make(bm);

            let moves = engine.startpos.gen();
            let game_state = engine.startpos.game_state(&moves, &engine.startstack);
            if game_state != GameState::Ongoing {
                break;
            }
        }
    }
}
