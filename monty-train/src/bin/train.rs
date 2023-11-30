use monty_engine::{PolicyNetwork, NetworkDims};
use monty_train::{gradient_batch, TrainingPosition, Rand};

use std::time::Instant;

const BATCH_SIZE: usize = 16_384;

fn data_from_bytes_with_lifetime(raw_bytes: &mut [u8]) -> &mut [TrainingPosition] {
    let src_size = std::mem::size_of_val(raw_bytes);
    let tgt_size = std::mem::size_of::<TrainingPosition>();

    assert!(
        src_size % tgt_size == 0,
        "Target type size does not divide slice size!"
    );

    let len = src_size / tgt_size;
    unsafe {
        std::slice::from_raw_parts_mut(raw_bytes.as_mut_ptr().cast(), len)
    }
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

    let mut lr = 0.1;

    for iteration in 1..=30 {
        println!("# [Training Epoch {iteration}]");
        train(threads, &mut policy, data, lr);

        if iteration % 10 == 0 {
            lr *= 0.1;
        }
    }

    policy.write_to_bin("policy.bin");
}

fn shuffle(data: &mut [TrainingPosition]) {
    let mut rng = Rand::with_seed();

    for _ in 0..data.len() * 4 {
        let idx1 = rng.rand_int() as usize % data.len();
        let idx2 = rng.rand_int() as usize % data.len();
        data.swap(idx1, idx2);
    }
}

fn train(threads: usize, policy: &mut PolicyNetwork, data: &[TrainingPosition], lr: f32) {
    let mut running_error = 0.0;

    for batch in data.chunks(BATCH_SIZE) {
        let mut grad = PolicyNetwork::boxed_and_zeroed();
        running_error += gradient_batch(threads, policy, &mut grad, batch);
        let adj = 2.0 / batch.len() as f32;
        update(policy, &grad, adj, lr);
    }

    println!("> Running Loss: {}", running_error / data.len() as f32);
}

fn update(policy: &mut PolicyNetwork, grad: &PolicyNetwork, adj: f32, lr: f32) {
    for i in 0..NetworkDims::INDICES {
        for j in 0..NetworkDims::FEATURES {
            policy.weights[i][j] -= lr * adj * grad.weights[i][j];
        }
    }
}