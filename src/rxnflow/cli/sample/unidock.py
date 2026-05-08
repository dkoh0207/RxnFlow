import click


@click.command("unidock", help="Inference sampling from trained UniDock-trained models.")
@click.option(
    "-m", "--model-path", "--model_path", "model_path", required=True, type=click.Path(), help="Model Checkpoint Path"
)
@click.option(
    "-n", "--num-samples", "--num_samples", "num_samples", required=True, type=int, help="Number of Samples"
)
@click.option(
    "-o",
    "--out-path",
    "--out_path",
    "out_path",
    required=True,
    type=click.Path(),
    help="Output Path (.csv | .smi). If csv, the docking score is calculated.",
)
@click.option(
    "--env-dir",
    "--env_dir",
    "env_dir",
    type=click.Path(),
    default=None,
    help="Environment Directory Path (overwrite training setting)",
)
@click.option(
    "--subsampling-ratio",
    "--subsampling_ratio",
    "subsampling_ratio",
    type=float,
    default=0.1,
    show_default=True,
    help="Action Subsampling Ratio. Memory-efficiency trade-off (Higher ratio increase samplinge efficiency; default: 0.1)",
)
@click.option("--cuda", is_flag=True, help="CUDA Acceleration")
@click.option("-p", "--protein", type=click.Path(), default=None, help="Protein PDB Path (overwrite training setting)")
@click.option("-c", "--center", nargs=3, type=float, default=None, help="Pocket Center (--center X Y Z)")
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
    "-s",
    "--size",
    nargs=3,
    type=float,
    default=(22.5, 22.5, 22.5),
    show_default=True,
    help="Search Box Size (--size X Y Z)",
)
@click.option(
    "--initial-scaffold",
    "--initial_scaffold",
    "initial_scaffold",
    type=str,
    default=None,
    help="SMILES of an initial scaffold molecule. If set, every sampled trajectory starts from this "
    "molecule instead of the blank state.",
)
def unidock(
    model_path,
    num_samples,
    out_path,
    env_dir,
    subsampling_ratio,
    cuda,
    protein,
    center,
    ref_ligand,
    size,
    initial_scaffold,
):
    import os
    import tempfile
    import time
    from pathlib import Path

    from rxnflow.config import Config, init_empty
    from rxnflow.tasks.unidock_vina import VinaSampler

    ckpt_path = Path(model_path)

    # change config from training
    config = init_empty(Config())

    # most samplings are generated in multiples of 100. e.g., generate 1000 molecules
    # 100 molecules for each iteration.
    config.algo.num_from_policy = 100

    # low subsampling ratio: force exploration
    # high subsampling ratio: more exploitation
    config.algo.action_subsampling.sampling_ratio = subsampling_ratio
    config.algo.initial_scaffold = initial_scaffold

    if env_dir is not None:
        config.env_dir = env_dir
    if protein is not None:
        config.task.docking.protein_path = protein
    if ref_ligand is not None:
        config.task.docking.ref_ligand_path = ref_ligand
    if center:
        config.task.docking.center = list(center)
    if size is not None:
        config.task.docking.size = list(size)

    device = "cuda" if cuda else "cpu"
    save_reward = os.path.splitext(out_path)[1] == ".csv"

    # NOTE: Run
    with tempfile.TemporaryDirectory() as tempdir:
        config.log_dir = tempdir
        sampler = VinaSampler(config, ckpt_path, device)
        tick_st = time.time()
        res = sampler.sample(num_samples, calc_reward=save_reward)
        tick_end = time.time()
    print(f"Sampling: {tick_end - tick_st:.3f} sec")
    print(f"Generated Molecules: {len(res)}")
    if save_reward:
        with open(out_path, "w") as w:
            w.write(",SMILES,Vina\n")
            for idx, sample in enumerate(res):
                smiles = sample["smiles"]
                vina = sample["info"]["reward"][0] * -1
                w.write(f"sample{idx},{smiles},{vina:.3f}\n")
    else:
        with open(out_path, "w") as w:
            for idx, sample in enumerate(res):
                w.write(f"{sample['smiles']}\tsample{idx}\n")
