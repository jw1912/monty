use crate::{
    pop_lsb,
    state::{
        consts::{Piece, Side},
        moves::Move,
        position::Position,
    },
};

pub const INDICES: usize = 6 * 64;
pub const FEATURES: usize = 768;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PolicyNetwork {
    weights: [[f64; FEATURES]; INDICES],
    biases: [f64; INDICES],
}

pub static POLICY_NETWORK: PolicyNetwork =
    unsafe { std::mem::transmute(*include_bytes!("../../resources/policy.bin")) };

pub fn get_policy(mov: &Move, pos: &Position, params: &PolicyNetwork) -> f64 {
    let flip = if pos.stm() == Side::BLACK { 56 } else { 0 };
    let idx = mov.index(flip);

    let weights_ref = &params.weights[idx];
    let mut score = params.biases[idx];

    for piece in Piece::PAWN..=Piece::KING {
        let pc = 64 * (piece - 2);

        let mut our_bb = pos.piece(piece) & pos.piece(pos.stm());
        while our_bb > 0 {
            pop_lsb!(sq, our_bb);
            score += weights_ref[pc + usize::from(sq ^ flip)];
        }

        let mut opp_bb = pos.piece(piece) & pos.piece(pos.stm() ^ 1);
        while opp_bb > 0 {
            pop_lsb!(sq, opp_bb);
            score += weights_ref[384 + pc + usize::from(sq ^ flip)];
        }
    }

    score
}
