use crate::{TrainingPosition, Rand};

use monty_core::{GameState, Position, STARTPOS};
use monty_engine::{PolicyNetwork, TunableParams, Searcher};

use std::{fs::File, io::{BufWriter, Write}};

unsafe fn to_slice_with_lifetime<T, U>(slice: &[T]) -> &[U] {
    let src_size = std::mem::size_of_val(slice);
    let tgt_size = std::mem::size_of::<U>();

    assert!(
        src_size % tgt_size == 0,
        "Target type size does not divide slice size!"
    );

    let len = src_size / tgt_size;
    std::slice::from_raw_parts(slice.as_ptr().cast(), len)
}

fn write(data: &mut Vec<TrainingPosition>, output: &mut BufWriter<File>) {
    if data.is_empty() {
        return
    }

    let data_slice = unsafe { to_slice_with_lifetime(data) };

    output
        .write_all(data_slice)
        .expect("Nothing can go wrong in unsafe code!");

    data.clear();
}

pub struct DatagenThread<'a> {
    id: u32,
    rng: Rand,
    params: TunableParams,
    policy: &'a PolicyNetwork,
    positions: Vec<TrainingPosition>,
    skipped: usize,
    total: usize,
}

impl<'a> DatagenThread<'a> {
    pub fn new(params: TunableParams, policy: &'a PolicyNetwork) -> Self {
        let mut rng = Rand::with_seed();
        Self {
            id: rng.rand_int(),
            rng,
            params,
            policy,
            positions: Vec::new(),
            skipped: 0,
            total: 0,
        }
    }

    pub fn run(&mut self, num_positions: usize) {
        let position = Position::parse_fen(STARTPOS);

        let out_path = format!("monty-{}.data", self.id);
        let mut output = BufWriter::new(File::create(out_path.as_str()).expect("Provide a correct path!"));

        while self.total < num_positions {
            self.run_game(position, self.params.clone(), self.policy);

            let num_in_buffer = self.positions.len();
            if num_in_buffer > 2048 {
                write(&mut self.positions, &mut output);
                println!("thread {} count {}", self.id, self.total);
                self.positions.clear();
            }
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
            let moves = engine.startpos.gen::<true>();

            if moves.is_empty() {
                return;
            }

            let mov = moves[self.rng.rand_int() as usize % moves.len()];

            engine.startstack.push(engine.startpos.hash());
            engine.startpos.make(mov, None);
        }

        if engine.startpos.gen::<true>().is_empty() {
            return;
        }

        // play out game
        loop {
            let (bm, _) = engine.search(None, 128, false, false, &mut 0);

            // disallow positions with >106 moves
            if engine.tree[0].moves.len() <= 106 {
                let mut training_pos = TrainingPosition::new(engine.startpos);

                for mov in engine.tree[0].moves.iter() {
                    if mov.ptr() == -1 {
                        continue;
                    }

                    let child = &engine.tree[mov.ptr() as usize];
                    let visits = child.visits();

                    training_pos.push(mov, visits);
                }

                self.positions.push(training_pos);
                self.total += 1;
            } else {
                self.skipped += 1;
            }

            engine.startstack.push(engine.startpos.hash());
            engine.startpos.make(bm, None);

            let moves = engine.startpos.gen::<true>();
            let game_state = engine.startpos.game_state(&moves, &engine.startstack);
            if game_state != GameState::Ongoing {
                break;
            }
        }
    }
}
