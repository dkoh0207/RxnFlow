import click


@click.command("pocket", help="Pocket-conditional GFlowNet training (CrossDocked DB).")
@click.option(
    "--db",
    type=click.Path(),
    default="./data/experiments/CrossDocked2020/train_db.pt",
    show_default=True,
    help="Pocket DB Path",
)
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
    default=64,
    show_default=True,
    help="Batch Size. Memory-variance trade-off (default: 128)",
)
@click.option(
    "--subsampling-ratio",
    "--subsampling_ratio",
    "subsampling_ratio",
    type=float,
    default=0.02,
    show_default=True,
    help="Action Subsampling Ratio. Memory-variance trade-off (Smaller ratio increase variance; default: 0.02)",
)
@click.option("--wandb", "wandb_name", type=str, default=None, help="wandb job name")
@click.option("--debug", is_flag=True, help="For debugging option")
def pocket(db, env_dir, out_dir, num_iterations, batch_size, subsampling_ratio, wandb_name, debug):
    import wandb

    from rxnflow.config import Config, init_empty
    from rxnflow.tasks.multi_pocket import ProxyTrainer_MultiPocket

    config = init_empty(Config())
    config.env_dir = env_dir
    config.log_dir = out_dir

    config.num_training_steps = num_iterations
    config.print_every = 50
    config.checkpoint_every = 1000
    config.store_all_checkpoints = True
    config.num_workers_retrosynthesis = 4

    config.task.pocket_conditional.pocket_db = db
    config.task.pocket_conditional.proxy = ("TacoGFN_Reward", "QVina", "ZINCDock15M")

    # === GFN parameters === #
    config.cond.temperature.sample_dist = "uniform"
    config.cond.temperature.dist_params = [0.0, 64.0]

    # === Training parameters === #
    config.algo.train_random_action_prob = 0.2
    config.algo.sampling_tau = 0.9

    # pretrain -> more train and better regularization with dropout
    config.model.dropout = 0.1

    # training batch size & subsampling size
    # cost-variance trade-off parameters
    config.algo.num_from_policy = batch_size
    config.algo.action_subsampling.sampling_ratio = subsampling_ratio

    # training learning rate
    config.opt.learning_rate = 1e-4
    config.opt.lr_decay = 10_000
    config.algo.tb.Z_learning_rate = 1e-2
    config.algo.tb.Z_lr_decay = 20_000

    if debug:
        config.overwrite_existing_exp = True
        config.print_every = 1
    if wandb_name:
        wandb.init(project="rxnflow", name=wandb_name, group="pocket-conditional")

    trainer = ProxyTrainer_MultiPocket(config)
    trainer.run()
    trainer.terminate()
