use crate::state::{moves::Move, position::Position};

fn mvv_lva(mov: &Move, pos: &Position) -> i32 {
    8 * pos.get_pc(1 << mov.to()) as i32 - mov.moved() as i32
}

pub fn quiesce(pos: &Position, mut alpha: i32, beta: i32) -> i32 {
    let mut eval = pos.eval_cp();

    // stand-pat
    if eval >= beta {
        return eval;
    }

    alpha = alpha.max(eval);

    let mut caps = pos.gen::<false>();
    caps.sort_by_cached_key(|cap| mvv_lva(cap, pos));

    for &mov in caps.iter() {
        // static exchange eval pruning
        if !pos.see(&mov, 1) {
            continue;
        }

        let mut new = *pos;
        new.make(mov);

        let score = -quiesce(&new, -beta, -alpha);

        eval = eval.max(score);
        alpha = alpha.max(eval);

        if eval >= beta {
            break;
        }
    }

    eval
}