use bullet::{
    inputs, outputs, Activation, LocalSettings, LrScheduler, TrainerBuilder,
    TrainingSchedule, WdlScheduler,
};

const HIDDEN_SIZE: usize = 8;

pub fn train_value() {
    let mut trainer = TrainerBuilder::default()
        .single_perspective()
        .quantisations(&[255, 64])
        .input(inputs::Ataxx98)
        .output_buckets(outputs::Single)
        .feature_transformer(HIDDEN_SIZE)
        .activate(Activation::SCReLU)
        .add_layer(1)
        .build();

    let schedule = TrainingSchedule {
        net_id: "ataxx-value".to_string(),
        eval_scale: 400.0,
        batch_size: 16_384,
        batches_per_superbatch: 100,
        start_superbatch: 1,
        end_superbatch: 40,
        wdl_scheduler: WdlScheduler::Constant { value: 1.0 },
        lr_scheduler: LrScheduler::Step { start: 0.001, gamma: 0.1, step: 15 },
        save_rate: 10,
    };

    let settings = LocalSettings {
        threads: 4,
        data_file_paths: vec!["ataxx-value001.data"],
        output_directory: "checkpoints",
    };

    trainer.run(&schedule, &settings);

    println!("{}", 400.0 * trainer.eval("x5o/7/7/7/7/7/o5x x 0 1"));
    println!("{}", 400.0 * trainer.eval("5oo/7/x6/x6/7/7/o5x o 0 2"));
}
