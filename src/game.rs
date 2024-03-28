use crate::moves::{MoveList, MoveType};

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum GameState {
    #[default]
    Ongoing,
    Lost,
    Draw,
    Won,
}

pub trait GameRep: Clone + Default {
    type PolicyNet;
    type ValueNet;
    type Move: MoveType;
    const MAX_MOVES: usize;
    const STARTPOS: &'static str;

    fn stm(&self) -> usize;

    fn game_state(&self) -> GameState;

    fn make_move(&mut self, mov: Self::Move);

    fn gen_legal_moves(&self) -> MoveList<Self::Move>;

    fn set_policies(&self, policy: &Self::PolicyNet, moves: &mut MoveList<Self::Move>);

    fn get_value(&self, value: &Self::ValueNet) -> f32;

    fn from_fen(fen: &str) -> Self;

    fn conv_mov_to_str(&self, mov: Self::Move) -> &str;

    fn perft(&mut self, depth: usize) -> u64;
}
