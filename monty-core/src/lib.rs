mod activation;
mod attacks;
mod consts;
mod moves;
mod policy;
mod position;
mod value;
mod vector;

pub use activation::ReLU;
pub use consts::{Flag, Piece, Side};
pub use moves::{Move, MoveList};
pub use policy::{NetworkDims, PolicyNetwork, POLICY_NETWORK, PolicyVal};
pub use position::{FeatureList, Position, GameState, perft};
pub use value::Accumulator;
pub use vector::{Matrix, Vector};

pub const STARTPOS: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

pub fn cp_wdl(score: i32) -> f32 {
    1.0 / (1.0 + (-(score as f32) / (400.0)).exp())
}
