import click


@click.command(
    "few-shot-unidock",
    help="GFlowNet few-shot training for Vina-QED multi-objective optimization (fine-tune from pretrained).",
)
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
    "--pretrained-model",
    "--pretrained_model",
    "pretrained_model",
    type=str,
    default="qvina-unif-0-64",
    show_default=True,
    help="Pretrained Model Path",
)
@click.option(
    "-n",
    "--num-iterations",
    "--num_iterations",
    "num_iterations",
    type=int,
    default=1_000,
    show_default=True,
    help="Number of training iterations (default: 1,000)",
)
@click.option(
    "--subsampling-ratio",
    "--subsampling_ratio",
    "subsampling_ratio",
    type=float,
    default=0.04,
    show_default=True,
    help="Action Subsampling Ratio. Memory-variance trade-off (Smaller ratio increase variance; default: 0.04)",
)
@click.option("--wandb", "wandb_name", type=str, default=None, help="wandb job name")
@click.option("--debug", is_flag=True, help="For debugging option")
def few_shot_unidock(
    protein,
    ref_ligand,
    center,
    size,
    search_mode,
    env_dir,
    out_dir,
    pretrained_model,
    num_iterations,
    subsampling_ratio,
    wandb_name,
    debug,
):
    import wandb

    from rxnflow.config import Config, init_empty
    from rxnflow.tasks.unidock_vina_fewshot import VinaMOOTrainer_Fewshot
    from rxnflow.utils.download import download_pretrained_weight

    config = init_empty(Config())
    config.env_dir = env_dir
    config.log_dir = out_dir
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

    # === GFN parameters === #
    config.cond.temperature.sample_dist = "uniform"
    config.cond.temperature.dist_params = [0.0, 64.0]

    # set EMA factor
    config.algo.sampling_tau = 0.98

    # replay buffer
    config.replay.use = True
    config.replay.capacity = 64 * 200
    config.replay.warmup = 64 * 20
    config.replay.num_from_replay = 256 - 64  # batch size = 256

    if debug:
        config.overwrite_existing_exp = True
        config.print_every = 1
    if wandb_name:
        wandb.init(project="rxnflow", name=wandb_name, group="pocket-conditional")

    # post-construction mutation: search_mode is set on the task after trainer init
    trainer = VinaMOOTrainer_Fewshot(config)
    trainer.task.vina.search_mode = search_mode
    trainer.run()
    trainer.terminate()
