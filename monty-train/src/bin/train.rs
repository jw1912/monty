use monty_engine::PolicyNetwork;
use monty_train::{gradient_batch, TrainingPosition, Rand};

fn main() {

}

fn run_training(threads: usize, data: &[TrainingPosition]) {
    let mut policy = PolicyNetwork::boxed_and_zeroed();

    for iteration in 1..=64 {
        println!("# [Training]");
        train(threads, &mut policy, data);

        policy.write_to_bin(format!("policy-{iteration}.bin").as_str());
    }
}

const DATAGEN_SIZE: usize = 16_384;
const BATCH_SIZE: usize = 1_024;
const LR: f32 = 1.0;

fn shuffle(data: &mut Vec<TrainingPosition>) {
    let mut rng = Rand::with_seed();

    for _ in 0..data.len() * 4 {
        let idx1 = rng.rand_int() as usize % data.len();
        let idx2 = rng.rand_int() as usize % data.len();
        data.swap(idx1, idx2);
    }
}

fn train(threads: usize, policy: &mut PolicyNetwork, data: &[TrainingPosition]) {
    let mut grad = PolicyNetwork::boxed_and_zeroed();
    let error = gradient_batch(threads, policy, &mut grad, data);
    println!("> Before Loss: {}", error / data.len() as f32);

    let mut running_error = 0.0;

    for batch in data.chunks(BATCH_SIZE) {
        let mut grad = PolicyNetwork::boxed_and_zeroed();
        running_error += gradient_batch(threads, policy, &mut grad, batch);
        let adj = LR / batch.len() as f32;
        update(policy, &grad, adj);
    }

    println!("> Running Loss: {}", running_error / data.len() as f32);

    let mut grad = PolicyNetwork::boxed_and_zeroed();
    let error = gradient_batch(threads, policy, &mut grad, &data);
    println!("> After Loss: {}", error / data.len() as f32);
}

fn update(policy: &mut PolicyNetwork, grad: &PolicyNetwork, adj: f32) {
    for (i, j) in policy.weights.iter_mut().zip(grad.weights.iter()) {
        for (a, b) in i.iter_mut().zip(j.iter()) {
            *a -= adj * *b;
        }
    }
}