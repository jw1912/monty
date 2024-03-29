mod board;
mod moves;
mod util;
mod value;

use crate::{GameRep, MoveType, UciLike};

pub use self::{board::Board, moves::Move};

const STARTPOS: &str = "x5o/7/7/7/7/7/o5x x 0 1";

pub struct Uai;
impl UciLike for Uai {
    type Game = Ataxx;
    const NAME: &'static str = "uai";
    const NEWGAME: &'static str = "uainewgame";
    const OK: &'static str = "uaiok";
    const FEN_STRING: &'static str = include_str!("../../resources/ataxx-fens.txt");

    fn options() {}
}

#[derive(Clone, Copy, Default)]
pub struct Ataxx {
    board: Board,
}

impl Ataxx {
    pub fn board(&self) -> &Board {
        &self.board
    }
}

impl GameRep for Ataxx {
    const STARTPOS: &'static str = STARTPOS;
    type Move = Move;
    type Policy = ();
    type Value = ();

    fn stm(&self) -> usize {
        self.board.stm()
    }

    fn tm_stm(&self) -> usize {
        self.board.stm() ^ 1
    }

    fn conv_mov_to_str(&self, mov: Self::Move) -> String {
        mov.uai()
    }

    fn from_fen(fen: &str) -> Self {
        Self { board: Board::from_fen(fen) }
    }

    fn game_state(&self) -> crate::GameState {
        self.board.game_state()
    }

    fn gen_legal_moves(&self) -> crate::MoveList<Self::Move> {
        self.board.movegen()
    }

    fn get_value(&self, _: &Self::Value) -> f32 {
        let out = value::ValueNetwork::eval(&self.board);

        1.0 / (1.0 + (-out as f32 / 400.0).exp())
    }

    fn set_policies(&self, _: &Self::Policy, moves: &mut crate::MoveList<Self::Move>) {
        let p = 1.0 / moves.len() as f32;

        for mov in moves.iter_mut() {
            mov.set_policy(p);
        }
    }

    fn make_move(&mut self, mov: Self::Move) {
        self.board.make(mov);
    }

    fn perft(&self, depth: usize) -> u64 {
        perft(&self.board, depth as u8)
    }
}

fn perft(board: &Board, depth: u8) -> u64 {
    if depth == 1 {
        return board.movegen_bulk(false);
    }

    let moves = board.movegen();
    let mut nodes = 0;

    for &mov in moves.iter() {
        let mut new = *board;

        if mov.is_pass() {
            continue;
        }

        new.make(mov);
        nodes += perft(&new, depth - 1);
    }

    nodes
}
