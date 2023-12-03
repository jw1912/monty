use monty_engine::{PolicyNetwork, NetworkDims, PolicyVal};
use monty_train::{gradient_batch, TrainingPosition, to_slice_with_lifetime, Rand};

use std::{fs::File, io::{BufReader, BufRead}};

const BATCH_SIZE: usize = 16_384;

fn main() {
    let mut args = std::env::args();
    args.next();

    let threads = args.next().unwrap().parse().unwrap();
    let data_path = args.next().unwrap();

    let file = File::open(data_path.clone()).unwrap();

    let mut policy = PolicyNetwork::boxed_and_zeroed();
    let mut rng = Rand::with_seed();
    for i in 0..NetworkDims::INDICES {
        for j in 0..NetworkDims::FEATURES {
            let mut val = [0.0; NetworkDims::NEURONS];
            for v in val.iter_mut() {
                *v = rng.rand_f32(0.2);
            }
            policy.weights[i][j] = PolicyVal::from_raw(val);
        }
    }

    for i in 0..NetworkDims::NEURONS {
        policy.outputs[i] = rng.rand_f32(0.1);
    }

    println!("# [Info]");
    println!(
        "> {} Positions",
        file.metadata().unwrap().len() / std::mem::size_of::<TrainingPosition>() as u64,
    );

    let mut lr = 0.001;
    let mut momentum = PolicyNetwork::boxed_and_zeroed();
    let mut velocity = PolicyNetwork::boxed_and_zeroed();

    for iteration in 1..=12 {
        println!("# [Training Epoch {iteration}]");
        train(threads, &mut policy, lr, &mut momentum, &mut velocity, data_path.as_str());

        if iteration % 8 == 0 {
            lr *= 0.1;
        }
        println!("{:?}", policy.hce);
        policy.write_to_bin("policy.bin");
    }
}

fn train(
    threads: usize,
    policy: &mut PolicyNetwork,
    lr: f32,
    momentum: &mut PolicyNetwork,
    velocity: &mut PolicyNetwork,
    path: &str,
) {
    let mut running_error = 0.0;
    let mut num = 0;

    let cap = 128 * BATCH_SIZE * std::mem::size_of::<TrainingPosition>();
    let file = File::open(path).unwrap();
    let mut loaded = BufReader::with_capacity(cap, file);

    while let Ok(buf) = loaded.fill_buf() {
        if buf.is_empty() {
            break;
        }

        let data = to_slice_with_lifetime(buf);

        for batch in data.chunks(BATCH_SIZE) {
            let mut grad = PolicyNetwork::boxed_and_zeroed();
            running_error += gradient_batch(threads, policy, &mut grad, batch);
            let adj = 2.0 / batch.len() as f32;
            update(policy, &grad, adj, lr, momentum, velocity);
        }

        num += data.len();
        let consumed = buf.len();
        loaded.consume(consumed);
    }

    println!("> Running Loss: {}", running_error / num as f32);
}

const B1: f32 = 0.9;
const B2: f32 = 0.999;

fn update(
    policy: &mut PolicyNetwork,
    grad: &PolicyNetwork,
    adj: f32,
    lr: f32,
    momentum: &mut PolicyNetwork,
    velocity: &mut PolicyNetwork,
) {
    for i in 0..NetworkDims::INDICES {
        for j in 0..NetworkDims::FEATURES {
            let g = adj * grad.weights[i][j];
            let m = &mut momentum.weights[i][j];
            let v = &mut velocity.weights[i][j];
            let p = &mut policy.weights[i][j];

            *m = B1 * *m + (1. - B1) * g;
            *v = B2 * *v + (1. - B2) * g * g;
            *p -= lr * *m / (v.sqrt() + 0.000_000_01);
        }
    }

    for i in 0..NetworkDims::NEURONS {
        let g = adj * grad.outputs[i];
        let m = &mut momentum.outputs[i];
        let v = &mut velocity.outputs[i];
        let p = &mut policy.outputs[i];

        *m = B1 * *m + (1. - B1) * g;
        *v = B2 * *v + (1. - B2) * g * g;
        *p -= lr * *m / (v.sqrt() + 0.000_000_01);
    }

    for i in 0..NetworkDims::HCE {
        let g = adj * grad.hce[i];
        let m = &mut momentum.hce[i];
        let v = &mut velocity.hce[i];
        let p = &mut policy.hce[i];

        *m = B1 * *m + (1. - B1) * g;
        *v = B2 * *v + (1. - B2) * g * g;
        *p -= lr * *m / (v.sqrt() + 0.000_000_01);
    }
}
