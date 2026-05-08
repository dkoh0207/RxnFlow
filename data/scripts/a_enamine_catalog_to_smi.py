import argparse
import multiprocessing
from pathlib import Path

from _a_refine import get_clean_smiles
from tqdm import tqdm


def main(block_path: str, save_block_path: str, num_cpus: int):
    block_file = Path(block_path)
    assert block_file.suffix == ".sdf"

    print("Read SDF Files")
    with block_file.open() as f:
        lines = f.readlines()

    # Enamine tag names drift across catalog releases: older SDFs use `<id>`,
    # newer ones use `<Catalog_ID>` (and similarly `<smiles>` vs `<SMILES>`).
    # Auto-detect from a candidate list so the script works regardless of release.
    def find_tag(candidates: list[str]) -> str:
        for tag in candidates:
            prefix = f">  <{tag}>"
            if any(line.startswith(prefix) for line in lines[:200_000]):
                return prefix
        raise RuntimeError(f"None of {candidates} found as SDF tags. Inspect the SDF and add the right tag name.")

    smiles_prefix = find_tag(["smiles", "SMILES", "Smiles"])
    id_prefix = find_tag(["Catalog_ID", "Catalog ID", "id", "ID", "idnumber"])
    print(f"Using SDF tags: smiles={smiles_prefix!r}, id={id_prefix!r}")

    smiles_list = [lines[i].strip() for i in tqdm(range(1, len(lines))) if lines[i - 1].startswith(smiles_prefix)]
    ids = [lines[i].strip() for i in tqdm(range(1, len(lines))) if lines[i - 1].startswith(id_prefix)]

    print(len(smiles_list), len(ids))
    assert len(smiles_list) == len(ids), "sdf file error, number of <smiles> and <id> should be matched"
    print("Including Mols:", len(smiles_list))

    print("Run Building Blocks...")
    clean_smiles_list = []
    for idx in tqdm(range(0, len(smiles_list), 10000)):
        chunk = smiles_list[idx : idx + 10000]
        with multiprocessing.Pool(num_cpus) as pool:
            results = pool.map(get_clean_smiles, chunk)
        clean_smiles_list.extend(results)

    with open(save_block_path, "w") as w:
        for smiles, id in zip(clean_smiles_list, ids, strict=True):
            if smiles is not None:
                w.write(f"{smiles}\t{id}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get clean building blocks")
    parser.add_argument(
        "-b", "--building_block_path", type=str, help="Path to input enamine building block file (.sdf)"
    )
    parser.add_argument(
        "-o",
        "--out_path",
        type=str,
        help="Path to output smiles file",
        default="./building_blocks/enamine_catalog.smi",
    )
    parser.add_argument("--cpu", type=int, help="Num Workers")
    args = parser.parse_args()

    main(args.building_block_path, args.out_path, args.cpu)
