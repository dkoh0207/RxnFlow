import click


@click.command("seh", help="SEH Proxy single-objective optimization.")
@click.option(
    "-o", "--out-dir", "--out_dir", "out_dir", required=True, type=click.Path(), help="Output directory"
)
@click.option(
    "-n",
    "--num-iterations",
    "--num_iterations",
    "num_iterations",
    type=int,
    default=10_000,
    show_default=True,
    help="Number of training iterations (default: 10,000)",
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
    "--subsampling-ratio",
    "--subsampling_ratio",
    "subsampling_ratio",
    type=float,
    default=0.01,
    show_default=True,
    help="Action Subsampling Ratio. Memory-variance trade-off (Smaller ratio increase variance; default: 0.01)",
)
@click.option("--wandb", "wandb_name", type=str, default=None, help="wandb job name")
@click.option("--debug", is_flag=True, help="For debugging option")
def seh(out_dir, num_iterations, env_dir, subsampling_ratio, wandb_name, debug):
    import wandb

    from rxnflow.config import Config, init_empty
    from rxnflow.tasks.seh import SEHTrainer

    config = init_empty(Config())
    config.env_dir = env_dir
    config.log_dir = out_dir
    config.print_every = 10
    config.num_training_steps = num_iterations
    config.algo.action_subsampling.sampling_ratio = subsampling_ratio

    config.opt.learning_rate = 1e-4
    config.opt.lr_decay = 2000
    config.algo.tb.Z_learning_rate = 1e-2
    config.algo.tb.Z_lr_decay = 5000

    if debug:
        config.overwrite_existing_exp = True
    if wandb_name is not None:
        wandb.init(project="rxnflow", name=wandb_name, group="seh")

    trainer = SEHTrainer(config)
    trainer.run()
    trainer.terminate()
