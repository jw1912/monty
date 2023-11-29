use monty_engine::PolicyNetwork;
use monty_train::{gradient_batch, TrainingPosition, Rand};

use std::time::Instant;

fn data_from_bytes_with_lifetime(raw_bytes: &mut [u8]) -> &mut [TrainingPosition] {
    unsafe { std::mem::transmute(raw_bytes) }
}

fn main() {
    let mut args = std::env::args();
    args.next();

    let threads = args.next().unwrap().parse().unwrap();
    let data_path = args.next().unwrap();

    let mut raw_bytes = std::fs::read(data_path).unwrap();
    let data = data_from_bytes_with_lifetime(&mut raw_bytes);

    let mut policy = PolicyNetwork::boxed_and_zeroed();

    println!("# [Info]");
    println!("> {} Positions", data.len());

    println!("# [Shuffling Data]");
    let time = Instant::now();
    shuffle(data);
    println!("> Took {:.2} seconds.", time.elapsed().as_secs_f32());

    for iteration in 1..=64 {
        println!("# [Training Epoch {iteration}]");
        train(threads, &mut policy, data);

        policy.write_to_bin(format!("policy-{iteration}.bin").as_str());
    }
}

const BATCH_SIZE: usize = 16_384;
const LR: f32 = 1.0;

fn shuffle(data: &mut [TrainingPosition]) {
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
    let error = gradient_batch(threads, policy, &mut grad, data);
    println!("> After Loss: {}", error / data.len() as f32);
}

fn update(policy: &mut PolicyNetwork, grad: &PolicyNetwork, adj: f32) {
    for (i, j) in policy.weights.iter_mut().zip(grad.weights.iter()) {
        for (a, b) in i.iter_mut().zip(j.iter()) {
            *a -= adj * *b;
        }
    }
}