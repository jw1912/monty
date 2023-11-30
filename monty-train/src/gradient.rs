use crate::TrainingPosition;

use monty_engine::PolicyNetwork;

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
    let feats = pos.get_features();

    let mut policies = Vec::with_capacity(pos.num_moves());
    let mut total = 0.0;
    let mut total_visits = 0;

    let flip = pos.board().flip_val();

    for training_mov in pos.moves() {
        let mov = training_mov.mov(pos.board());
        let visits = training_mov.visits();

        let mut score = PolicyNetwork::hce(&mov, pos.board());

        let pc = usize::from(mov.moved() - 2);
        let sq = 6 + usize::from(mov.to() ^ flip);

        assert!(pc < 6, "{pos:?}");

        for &feat in &feats {
            score += policy.weights[pc][feat];
            score += policy.weights[sq][feat];
        }

        score = score.exp();

        total += score;
        total_visits += visits;
        policies.push((mov, visits, score));
    }

    for (mov, visits, score) in policies {
        let pc = usize::from(mov.moved() - 2);
        let sq = 6 + usize::from(mov.to() ^ pos.board().flip_val());

        let ratio = score / total;

        let expected = visits as f32 / total_visits as f32;
        let err = ratio - expected;

        *error += err * err;

        let dp = (1.0 - ratio) / total;
        let adj = err * score * dp;

        assert!(!adj.is_nan() && !adj.is_infinite(), "{score} {visits} {dp} {adj} {total}");

        for &feat in &feats {
            grad.weights[pc][feat] += adj;
            grad.weights[sq][feat] += adj;
        }
    }
}