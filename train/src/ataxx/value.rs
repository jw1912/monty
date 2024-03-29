use bullet::{
    format::AtaxxBoard, inputs::InputType, outputs, Activation, LocalSettings, LrScheduler, TrainerBuilder,
    TrainingSchedule, WdlScheduler,
};

const HIDDEN_SIZE: usize = 8;
const PER_TUPLE: usize = 3usize.pow(4);
const NUM_TUPLES: usize = 36;

pub fn train_value() {
    let mut trainer = TrainerBuilder::default()
        .single_perspective()
        .quantisations(&[255, 64])
        .input(Ataxx2Tuples)
        .output_buckets(outputs::Single)
        .feature_transformer(HIDDEN_SIZE)
        .activate(Activation::SCReLU)
        .add_layer(1)
        .build();

    let schedule = TrainingSchedule {
        net_id: "ataxx-value004".to_string(),
        eval_scale: 400.0,
        batch_size: 16_384,
        batches_per_superbatch: 512,
        start_superbatch: 1,
        end_superbatch: 40,
        wdl_scheduler: WdlScheduler::Constant { value: 0.5 },
        lr_scheduler: LrScheduler::Step { start: 0.001, gamma: 0.1, step: 15 },
        save_rate: 10,
    };

    let settings = LocalSettings {
        threads: 4,
        data_file_paths: vec!["data/ataxx/ataxx-value003.data"],
        output_directory: "checkpoints",
    };

    trainer.run(&schedule, &settings);

    println!("{}", 400.0 * trainer.eval("x5o/7/7/7/7/7/o5x x 0 1"));
    println!("{}", 400.0 * trainer.eval("5oo/7/x6/x6/7/7/o5x o 0 2"));
}

#[derive(Clone, Copy, Default)]
pub struct Ataxx2Tuples;
impl InputType for Ataxx2Tuples {
    type RequiredDataType = AtaxxBoard;
    type FeatureIter = ThisIterator;

    fn max_active_inputs(&self) -> usize {
        NUM_TUPLES
    }

    fn buckets(&self) -> usize {
        1
    }

    fn inputs(&self) -> usize {
        self.max_active_inputs() * PER_TUPLE
    }

    fn feature_iter(&self, pos: &Self::RequiredDataType) -> Self::FeatureIter {
        let mut res = [(0, 0); NUM_TUPLES];

        let [boys, opps, _] = pos.bbs();

        for i in 0..6 {
            for j in 0..6 {
                const POWERS: [usize; 4] = [1, 3, 9, 27];
                const MASK: u64 = 0b0001_1000_0011;

                let tuple = 6 * i + j;
                let mut feat = PER_TUPLE * tuple;

                let offset = 7 * i + j;
                let mut b = (boys >> offset) & MASK;
                let mut o = (opps >> offset) & MASK;

                while b > 0 {
                    let mut sq = b.trailing_zeros() as usize;
                    if sq > 6 {
                        sq -= 5;
                    }

                    feat += POWERS[sq];

                    b &= b - 1;
                }

                while o > 0 {
                    let mut sq = o.trailing_zeros() as usize;
                    if sq > 6 {
                        sq -= 5;
                    }

                    feat += 2 * POWERS[sq];

                    o &= o - 1;
                }

                res[tuple] = (feat, feat);
            }
        }

        ThisIterator { inner: res, index: 0 }
    }
}

pub struct ThisIterator {
    inner: [(usize, usize); NUM_TUPLES],
    index: usize,
}

impl Iterator for ThisIterator {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= NUM_TUPLES {
            return None;
        }

        let res = self.inner[self.index];
        self.index += 1;
        Some(res)
    }
}
