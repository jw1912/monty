use crate::{consts::Flag, moves::Move};

pub fn get_policy(mov: &Move) -> f64 {
    let val = if mov.flag() & Flag::CAP > 0 {
        2f64
    } else {
        0f64
    };

    val.exp()
}