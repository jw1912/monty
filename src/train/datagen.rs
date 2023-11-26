use crate::{
    search::{mcts::Searcher, params::TunableParams, policy::PolicyNetwork},
    state::position::{GameState, Position},
    train::rng::Rand,
};

use std::time::{SystemTime, UNIX_EPOCH};

pub struct DatagenThread<'a> {
    id: u32,
    rng: Rand,
    games: u64,
    fens: u64,
    params: TunableParams,
    policy: &'a PolicyNetwork,
}

impl<'a> DatagenThread<'a> {
    pub fn new(params: TunableParams, policy: &'a PolicyNetwork) -> Self {
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Guaranteed increasing.")
            .as_micros() as u32;

        let res = Self {
            id: seed,
            rng: Rand::new(seed),
            games: 0,
            fens: 0,
            params,
            policy,
        };

        println!("thread id {} created", res.id);
        res
    }

    pub fn run_game(
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
