mod datagen;
mod rng;

use crate::{search::{policy::PolicyNetwork, params::TunableParams}, state::{consts::{Side, Piece}, position::Position}, pop_lsb};
use self::{datagen::{run_datagen, TrainingPosition}, rng::Rand};

pub fn run_training(params: TunableParams, policy: &mut PolicyNetwork) {
    for iteration in 1..=64 {
        println!("# [Generating Data]");
        let mut data = run_datagen(1_000_000, params.clone(), policy);

        println!("# [Shuffling]");
        shuffle(&mut data);

        println!("# [Training]");
        train(policy, data);

        policy.write_to_bin(format!("policy-{iteration}.bin").as_str());
    }
}

fn shuffle(data: &mut Vec<TrainingPosition>) {
    let mut rng = Rand::new(101298019);

    for _ in 0..4_000_000 {
        let idx1 = rng.rand_int() as usize % data.len();
        let idx2 = rng.rand_int() as usize % data.len();
        data.swap(idx1, idx2);
    }
}

fn train(policy: &mut PolicyNetwork, data: Vec<TrainingPosition>) {
    let mut grad = PolicyNetwork::boxed_and_zeroed();

    for (i, batch) in data.chunks(16_384).enumerate() {
        println!("# [Batch {}]", i + 1);
        gradient_batch(policy, &mut grad, batch);
    }

    update(policy, &grad);
}

fn gradient_batch(policy: &PolicyNetwork, grad: &mut PolicyNetwork, batch: &[TrainingPosition]) {
    let threads = 6;
    let size = batch.len() / threads;

    std::thread::scope(|s| {
        batch
            .chunks(size)
            .map(|chunk| {
                s.spawn(move || {
                    let mut inner_grad = PolicyNetwork::boxed_and_zeroed();
                    for pos in chunk {
                        update_single_grad(pos, policy, &mut inner_grad);
                    }
                    inner_grad
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|p| p.join().unwrap())
            .for_each(|part| *grad += &part);
    });
}

fn get_features(pos: &Position) -> Vec<usize> {
    let mut res = Vec::with_capacity(pos.occ().count_ones() as usize);
    let flip = if pos.stm() == Side::BLACK { 56 } else { 0 };

    // bias is just an always-present feature
    res.push(768);

    for piece in Piece::PAWN..=Piece::KING {
        let pc = 64 * (piece - 2);

        let mut our_bb = pos.piece(piece) & pos.piece(pos.stm());
        while our_bb > 0 {
            pop_lsb!(sq, our_bb);
            res.push(pc + usize::from(sq ^ flip));
        }

        let mut opp_bb = pos.piece(piece) & pos.piece(pos.stm() ^ 1);
        while opp_bb > 0 {
            pop_lsb!(sq, opp_bb);
            res.push(384 + pc + usize::from(sq ^ flip));
        }
    }

    res
}

fn update_single_grad(pos: &TrainingPosition, policy: &PolicyNetwork, grad: &mut PolicyNetwork) {
    let feats = get_features(&pos.position);

    let mut policies = Vec::with_capacity(pos.moves.len());
    let mut total = 0.0;

    for (idx, _) in &pos.moves {
        let mut score = 0.0;
        for &feat in &feats {
            score += policy.weights[*idx][feat];
        }

        score = score.exp();

        total += score;
        policies.push(score);
    }

    for ((idx, expected), score) in pos.moves.iter().zip(policies.iter()) {
        let err = score / total - expected;
        let adj = 2.0 * err * score * (total - score) / total.powi(2);

        for &feat in &feats {
            grad.weights[*idx][feat] += adj;
        }
    }
}

fn update(policy: &mut PolicyNetwork, grad: &PolicyNetwork) {
    for (i, j) in policy.weights.iter_mut().zip(grad.weights.iter()) {
        for (a, b) in i.iter_mut().zip(j.iter()) {
            *a -= 0.001 * *b;
        }
    }
}
