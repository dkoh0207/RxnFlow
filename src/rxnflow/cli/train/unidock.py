import click


@click.command("unidock", help="Vina optimization with GPU-accelerated UniDock.")
@click.option("-p", "--protein", required=True, type=click.Path(), help="Protein PDB Path")
@click.option(
    "-l",
    "--ref-ligand",
    "--ref_ligand",
    "ref_ligand",
    type=click.Path(),
    default=None,
    help="Reference Ligand Path (required if center is missing)",
)
@click.option(
    "-c", "--center", nargs=3, type=float, default=None, help="Pocket Center (--center X Y Z)"
)
@click.option(
    "-s",
    "--size",
    nargs=3,
    type=float,
    default=(22.5, 22.5, 22.5),
    show_default=True,
    help="Search Box Size (--size X Y Z)",
)
@click.option(
    "--search-mode",
    "--search_mode",
    "search_mode",
    type=click.Choice(["fast", "balance", "detail"]),
    default="fast",
    show_default=True,
    help="UniDock Search Mode",
)
@click.option(
    "--filter",
    "filter_",
    type=click.Choice(["null", "lipinski", "veber"]),
    default="lipinski",
    show_default=True,
    help="Drug Filter",
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
    default=1000,
    show_default=True,
    help="Number of training iterations (64 molecules for each iterations; default: 1000)",
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
@click.option(
    "--pretrained-model",
    "--pretrained_model",
    "pretrained_model",
    type=str,
    default=None,
    help="Pretrained model path",
)
@click.option("--wandb", "wandb_name", type=str, default=None, help="wandb job name")
@click.option("--debug", is_flag=True, help="For debugging option")
def unidock(
    protein,
    ref_ligand,
    center,
    size,
    search_mode,
    filter_,
    env_dir,
    out_dir,
    num_iterations,
    subsampling_ratio,
    pretrained_model,
    wandb_name,
    debug,
):
    import wandb

    from rxnflow.config import Config, init_empty
    from rxnflow.tasks.unidock_vina import VinaTrainer
    from rxnflow.utils.download import download_pretrained_weight

    config = init_empty(Config())
    config.env_dir = env_dir
    config.log_dir = out_dir

    if pretrained_model is not None:
        config.pretrained_model_path = str(download_pretrained_weight(pretrained_model))

    config.print_every = 1
    config.num_training_steps = num_iterations
    config.algo.num_from_policy = 64
    config.algo.action_subsampling.sampling_ratio = subsampling_ratio

    # docking info
    config.task.docking.protein_path = protein
    config.task.docking.ref_ligand_path = ref_ligand
    config.task.docking.center = list(center) if center else None
    config.task.docking.size = list(size)

    # drug filter
    config.task.constraint.rule = filter_

    # set EMA factor
    if pretrained_model is None:
        config.algo.sampling_tau = 0.9
    else:
        config.algo.sampling_tau = 0.98

    # replay buffer
    config.replay.use = True
    config.replay.capacity = 64 * 200
    config.replay.warmup = 64 * 20
    config.replay.num_from_replay = 256 - 64  # batch size = 256

    if debug:
        config.overwrite_existing_exp = True
    if wandb_name is not None:
        wandb.init(project="rxnflow", name=wandb_name, group="unidock")

    trainer = VinaTrainer(config)
    trainer.task.vina.search_mode = search_mode  # set search mode
    trainer.run()
    trainer.terminate()
