mod attacks;
mod board;
mod consts;
mod frc;
mod moves;
mod policy;
mod qsearch;
mod value;

use crate::{comm::UciLike, game::{GameRep, GameState}, moves::{MoveList, MoveType}};

use self::{board::Board, frc::Castling, moves::Move, policy::PolicyNetwork, qsearch::quiesce, value::ValueNetwork};

pub use self::{policy::POLICY_NETWORK, value::NNUE};

const STARTPOS: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

#[derive(Clone)]
pub struct Chess {
    board: Board,
    castling: Castling,
    stack: Vec<u64>,
}

impl Default for Chess {
    fn default() -> Self {
        let mut castling = Castling::default();
        let board = Board::parse_fen(STARTPOS, &mut castling);

        Self { board, castling, stack: Vec::new() }
    }
}

pub struct Uci;
impl UciLike for Uci {
    const NAME: &'static str = "uci";
    const NEWGAME: &'static str = "ucinewgame";
    const OK: &'static str = "uciok";

    type Game = Chess;

    fn options() {
        println!("option name UCI_Chess960 type check default false");
    }
}

impl GameRep for Chess {
    type Move = Move;
    type Policy = PolicyNetwork;
    type Value = ValueNetwork;

    const STARTPOS: &'static str = STARTPOS;

    fn conv_mov_to_str(&self, mov: Self::Move) -> String {
        mov.to_uci(&self.castling)
    }

    fn from_fen(fen: &str) -> Self {
        let mut castling = Castling::default();
        let board = Board::parse_fen(fen, &mut castling);

        Self { board, castling, stack: Vec::new() }
    }

    fn gen_legal_moves(&self) -> MoveList<Move> {
        self.board.gen::<true>(&self.castling)
    }

    fn game_state(&self) -> GameState {
        let moves = self.gen_legal_moves();
        self.board.game_state(&moves, &self.stack)
    }

    fn make_move(&mut self, mov: Self::Move) {
        self.stack.push(self.board.hash());
        self.board.make(mov, None, &self.castling);
    }

    fn stm(&self) -> usize {
        self.board.stm()
    }

    fn get_value(&self, _: &Self::Value) -> f32 {
        let accs = self.board.get_accs();
        let qs = quiesce(&self.board, &self.castling, &accs, -30_000, 30_000);
        1.0 / (1.0 + (-(qs as f32) / (400.0)).exp())
    }

    fn set_policies(&self, policy: &Self::Policy, moves: &mut MoveList<Move>) {
        let mut total = 0.0;
        let mut max = -1000.0;
        let mut floats = [0.0; 256];
        let feats = self.board.get_features();

        for (i, mov) in moves.iter_mut().enumerate() {
            floats[i] = PolicyNetwork::get(mov, &self.board, policy, &feats);
            if floats[i] > max {
                max = floats[i];
            }
        }

        for (i, _) in moves.iter_mut().enumerate() {
            floats[i] = (floats[i] - max).exp();
            total += floats[i];
        }

        for (i, mov) in moves.iter_mut().enumerate() {
            mov.set_policy(floats[i] / total);
        }
    }

    fn perft(&self, depth: usize) -> u64 {
        perft::<true, true>(&self.board, depth as u8, &self.castling)
    }
}

fn perft<const ROOT: bool, const BULK: bool>(pos: &Board, depth: u8, castling: &Castling) -> u64 {
    let moves = pos.gen::<true>(castling);

    if BULK && !ROOT && depth == 1 {
        return moves.len() as u64;
    }

    let mut positions = 0;
    let leaf = depth == 1;

    for m_idx in 0..moves.len() {
        let mut tmp = *pos;
        tmp.make(moves[m_idx], None, castling);

        let num = if !BULK && leaf {
            1
        } else {
            perft::<false, BULK>(&tmp, depth - 1, castling)
        };
        positions += num;

        if ROOT {
            println!("{}: {num}", moves[m_idx].to_uci(castling));
        }
    }

    positions
}
