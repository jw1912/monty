use crate::TrainingPosition;

use monty_core::{Flag, PolicyNetwork, PolicyVal};

pub fn gradient_batch(threads: usize, policy: &PolicyNetwork, grad: &mut PolicyNetwork, batch: &[TrainingPosition]) -> f32 {
    let size = (batch.len() / threads).max(1);
    let mut errors = vec![0.0; threads];

    std::thread::scope(|s| {
        batch
            .chunks(size)
            .zip(errors.iter_mut())
            .map(|(chunk, err)| {
                s.spawn(move || {
                    let mut inner_grad = PolicyNetwork::boxed_and_zeroed();
                    for pos in chunk {
                        update_single_grad(pos, policy, &mut inner_grad, err);
                    }
                    inner_grad
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|p| p.join().unwrap())
            .for_each(|part| *grad += &part);
    });

    errors.iter().sum::<f32>()
}

fn update_single_grad(pos: &TrainingPosition, policy: &PolicyNetwork, grad: &mut PolicyNetwork, error: &mut f32) {
    let feats = pos.board().get_features();

    let mut policies = Vec::with_capacity(pos.num_moves());
    let mut total = 0.0;
    let mut total_visits = 0;
    let mut max = -1000.0;

    let flip = pos.board().flip_val();

    for training_mov in pos.moves() {
        let mov = training_mov.mov(pos.board());
        let visits = training_mov.visits();
        let idx = mov.index(flip);
        let sq = 6 + (idx % 64);
        let pc = idx / 64;

        let mut sq_hidden = PolicyVal::default();
        for &feat in feats.iter() {
            sq_hidden += policy.weights[sq][feat];
        }

        let mut pc_hidden = PolicyVal::default();
        for &feat in feats.iter() {
            pc_hidden += policy.weights[pc][feat];
        }

        let score = pc_hidden.out(&sq_hidden) + policy.hce(&mov, pos.board());

        if score > max {
            max = score;
        }

        total_visits += visits;
        policies.push((mov, visits, score, pc_hidden, sq_hidden));
    }

    for (_, _, score, _, _) in policies.iter_mut() {
        *score = (*score - max).exp();
        total += *score;
    }

    for (mov, visits, score, pc_hidden, sq_hidden) in policies {
        let idx = mov.index(flip);
        let sq = 6 + (idx % 64);
        let pc = idx / 64;

        let ratio = score / total;

        let expected = visits as f32 / total_visits as f32;
        let err = ratio - expected;

        *error += err * err;

        let factor = err * ratio * (1.0 - ratio);

        let pc_adj = factor * sq_hidden.activate() * pc_hidden.derivative();
        for &feat in feats.iter() {
            grad.weights[pc][feat] += pc_adj;
        }

        let sq_adj = factor * pc_hidden.activate() * sq_hidden.derivative();
        for &feat in feats.iter() {
            grad.weights[sq][feat] += sq_adj;
        }

        if pos.board().see(&mov, -108) {
            grad.hce[0] += factor;
        }

        if [Flag::QPR, Flag::QPC].contains(&mov.flag()) {
            grad.hce[1] += factor;
        }

        if mov.is_capture() {
            grad.hce[2] += factor;

            let diff = pos.board().get_pc(1 << mov.to()) as i32 - i32::from(mov.moved());
            grad.hce[3] += factor * diff as f32;
        }
    }
}