mod datagen;
mod rng;

use crate::search::{policy::PolicyNetwork, params::TunableParams};
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

fn update_single_grad(pos: &TrainingPosition, policy: &PolicyNetwork, grad: &mut PolicyNetwork) {

}

fn update(policy: &mut PolicyNetwork, grad: &PolicyNetwork) {
    for (i, j) in policy.weights.iter_mut().zip(grad.weights.iter()) {
        for (a, b) in i.iter_mut().zip(j.iter()) {
            *a -= 0.001 * *b;
        }
    }
}
