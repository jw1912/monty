use crate::{consts::Flag, moves::Move, params::TunableParams, position::Position};

fn safe_from_pawns(mov: &Move, threats: u64) -> bool {
    threats & (1 << mov.to()) == 0
}

pub fn get_policy(mov: &Move, pos: &Position, threats: u64, params: &TunableParams) -> f64 {
    let mut score = 0.0;

    if [Flag::QPR, Flag::QPC].contains(&mov.flag()) {
        score += params.promo();
    }

    if mov.flag() & Flag::CAP > 0 {
        score += params.cap();

        if usize::from(mov.moved()) < pos.get_pc(1 << mov.to()) {
            score += params.mvv_lva();
        }
    }

    if safe_from_pawns(mov, threats) {
        score += params.pawn_threat();
    }

    score
}