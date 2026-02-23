import os
import hashlib
import argparse
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import pandas as pd


def add_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--stl_dir",
        type=str,
        default=None,
        help="Directory containing STL files (required only for initial metadata creation)",
    )


def get_metadata(stl_dir=None, **kwargs):
    if stl_dir is None:
        raise ValueError("--stl_dir is required for initial metadata creation")

    stl_files = sorted([f for f in os.listdir(stl_dir) if f.lower().endswith(".stl")])
    if not stl_files:
        raise ValueError(f"No STL files found in {stl_dir}")

    records = []
    for fname in tqdm(stl_files, desc="Computing SHA256"):
        fpath = os.path.join(stl_dir, fname)
        with open(fpath, "rb") as f:
            sha256 = hashlib.sha256(f.read()).hexdigest()
        records.append(
            {
                "sha256": sha256,
                "local_path": fname,
                "aesthetic_score": 10.0,
            }
        )

    return pd.DataFrame(records)


def foreach_instance(
    metadata,
    output_dir,
    func,
    max_workers=None,
    desc="Processing objects",
    no_file=False,
):
    records = []
    if max_workers is None or max_workers <= 0:
        max_workers = os.cpu_count()

    try:
        with (
            ThreadPoolExecutor(max_workers=max_workers) as executor,
            tqdm(total=len(metadata), desc=desc) as pbar,
        ):

            def worker(metadatum):
                try:
                    if no_file:
                        record = func(None, metadatum)
                    else:
                        local_path = metadatum["local_path"]
                        file = os.path.join(output_dir, local_path)
                        record = func(file, metadatum)

                    if record is not None:
                        records.append(record)
                    pbar.update()
                except Exception as e:
                    print(
                        f"Error processing object {metadatum.get('sha256', 'unknown')}: {e}"
                    )
                    pbar.update()

            for metadatum in metadata.to_dict("records"):
                executor.submit(worker, metadatum)

            executor.shutdown(wait=True)
    except Exception as e:
        print(f"Error happened during processing: {e}")

    return pd.DataFrame.from_records(records)
