import click


@click.command("zero-shot", help="Zero-shot pocket-conditional sampling with the QED-Docking proxy model.")
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
@click.option("-c", "--center", nargs=3, type=float, default=None, help="Pocket Center (--center X Y Z)")
@click.option(
    "--model-path",
    "--model_path",
    "model_path",
    type=str,
    default="qvina-unif-0-64",
    show_default=True,
    help="Checkpoint Path (model name or filesystem path)",
)
@click.option(
    "-n",
    "--num-samples",
    "--num_samples",
    "num_samples",
    type=int,
    default=100,
    show_default=True,
    help="Number of Samples (default: 100)",
)
@click.option(
    "-o", "--out-path", "--out_path", "out_path", required=True, type=click.Path(), help="Output Path (.csv | .smi)"
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
    default=0.1,
    show_default=True,
    help="Action Subsampling Ratio. Memory-variance trade-off (Smaller ratio increase variance; default: 0.1)",
)
@click.option(
    "--temperature",
    type=str,
    default="uniform-16-64",
    show_default=True,
    help="temperature setting (e.g., uniform-16-64(default), uniform-32-64, ...)",
)
@click.option("--cuda", is_flag=True, help="CUDA Acceleration")
@click.option("--seed", type=int, default=1, show_default=True, help="seed")
def zero_shot(
    protein,
    ref_ligand,
    center,
    model_path,
    num_samples,
    out_path,
    env_dir,
    subsampling_ratio,
    temperature,
    cuda,
    seed,
):
    import os
    import time

    from rxnflow.cli._utils import parse_temperature
    from rxnflow.config import Config, init_empty
    from rxnflow.tasks.multi_pocket import ProxySampler
    from rxnflow.utils.download import download_pretrained_weight

    # change config from training
    config = init_empty(Config())
    config.seed = seed
    config.env_dir = env_dir
    config.algo.num_from_policy = 100
    config.algo.action_subsampling.sampling_ratio = subsampling_ratio

    device = "cuda" if cuda else "cpu"
    save_reward = os.path.splitext(out_path)[1] == ".csv"

    # create sampler
    ckpt = download_pretrained_weight(model_path)
    sampler = ProxySampler(config, ckpt, device)
    sample_dist, dist_params = parse_temperature(temperature)
    sampler.update_temperature(sample_dist, dist_params)

    # set binding site
    sampler.set_pocket(protein, list(center) if center else None, ref_ligand)

    # run
    tick = time.time()
    res = sampler.sample(num_samples, calc_reward=save_reward)
    print(f"Sampling: {time.time() - tick:.3f} sec")
    print(f"Generated Molecules: {len(res)}")
    if save_reward:
        with open(out_path, "w") as w:
            w.write(",SMILES,QED,Proxy\n")
            for idx, sample in enumerate(res):
                smiles = sample["smiles"]
                qed = sample["info"]["reward_qed"]
                proxy = sample["info"]["reward_vina"]
                w.write(f"sample{idx},{smiles},{qed:.3f},{proxy:.3f}\n")
    else:
        with open(out_path, "w") as w:
            for idx, sample in enumerate(res):
                w.write(f"{sample['smiles']}\tsample{idx}\n")
