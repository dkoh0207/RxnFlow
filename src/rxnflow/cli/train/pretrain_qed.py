import click


@click.command("pretrain-qed", help="QED pretraining for drug-likeness objective.")
@click.option(
    "--env-dir",
    "--env_dir",
    "env_dir",
    type=click.Path(),
    default="./data/envs/catalog",
    show_default=True,
    help="Environment Directory Path",
)
@click.option(
    "-o", "--out-dir", "--out_dir", "out_dir", required=True, type=click.Path(), help="Output directory"
)
@click.option(
    "--temperature",
    type=str,
    default="uniform-0-64",
    show_default=True,
    help="temperature setting (e.g., constant-32 ; uniform-0-64(default))",
)
@click.option(
    "-n",
    "--num-iterations",
    "--num_iterations",
    "num_iterations",
    type=int,
    default=50_000,
    show_default=True,
    help="Number of training iterations (default: 50,000)",
)
@click.option(
    "--batch-size",
    "--batch_size",
    "batch_size",
    type=int,
    default=128,
    show_default=True,
    help="Batch Size. Memory-variance trade-off (default: 128)",
)
@click.option(
    "--subsampling-ratio",
    "--subsampling_ratio",
    "subsampling_ratio",
    type=float,
    default=0.05,
    show_default=True,
    help="Action Subsampling Ratio. Memory-variance trade-off (Smaller ratio increase variance; default: 0.05)",
)
@click.option("--wandb", "wandb_name", type=str, default=None, help="wandb job name")
@click.option("--debug", is_flag=True, help="For debugging option")
def pretrain_qed(env_dir, out_dir, temperature, num_iterations, batch_size, subsampling_ratio, wandb_name, debug):
    import wandb

    from rxnflow.cli._utils import parse_temperature
    from rxnflow.config import Config, init_empty
    from rxnflow.tasks.qed import QEDTrainer

    config = init_empty(Config())
    config.env_dir = env_dir
    config.log_dir = out_dir

    config.num_training_steps = num_iterations
    config.print_every = 50
    config.checkpoint_every = 500
    config.store_all_checkpoints = True
    config.num_workers_retrosynthesis = 4

    # === GFN parameters === #
    sample_dist, dist_params = parse_temperature(temperature)
    config.cond.temperature.sample_dist = sample_dist
    config.cond.temperature.dist_params = dist_params

    # === Training parameters === #
    # we set high random action prob
    # so, we do not use Double-GFN
    config.algo.train_random_action_prob = 0.5
    config.algo.sampling_tau = 0.0

    # pretrain -> more train and better regularization with dropout
    config.model.dropout = 0.1

    # training batch size & subsampling size
    # cost-variance trade-off parameters
    config.algo.num_from_policy = batch_size
    config.algo.action_subsampling.sampling_ratio = subsampling_ratio

    # replay buffer
    # each training batch: 128 mols from policy and 128 mols from buffer
    config.replay.use = True
    config.replay.warmup = batch_size * 10
    config.replay.capacity = batch_size * 200

    # training learning rate
    config.opt.learning_rate = 1e-4
    config.opt.lr_decay = 10_000
    config.algo.tb.Z_learning_rate = 1e-2
    config.algo.tb.Z_lr_decay = 20_000

    if debug:
        config.overwrite_existing_exp = True
        config.print_every = 1
    if wandb_name is not None:
        wandb.init(project="rxnflow", name=wandb_name, group="qed-pretrain")

    trainer = QEDTrainer(config)
    trainer.run()
    trainer.terminate()
