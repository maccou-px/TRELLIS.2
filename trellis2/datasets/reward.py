from typing import *
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from ..modules import sparse as sp
from ..utils.data_utils import load_balanced_group_indices
import copy


class RewardDataset(Dataset):
    """
    Dataset for predicting scalar values from pre-extracted latents.

    Args:
        data_dir (str): Directory containing {sha256}.npz files
        sim_csv (str): Path to the simulation results CSV
            (columns include raw_filename, target_keys).
        latent_meta_csv (str): Path to the latent metadata CSV
            (columns: sha256, local_path)
        target_keys (list[str]): Columns from sim CSV to predict.
        normalize_targets (bool): Z-normalize scalar targets.
    """

    def __init__(
        self,
        data_dir: str,
        *,
        sim_csv: str,
        latent_meta_csv: str,
        target_keys: List[str] = ['Cd', 'Cl'],
        normalize_targets: bool = False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.target_keys = target_keys
        self.normalize_targets = normalize_targets
        self.value_range = (-1, 1)

        # ---- Load & join ----
        sim_df = pd.read_csv(sim_csv)
        latent_meta = pd.read_csv(latent_meta_csv)

        # filename → sha256
        latent_meta['join_key'] = latent_meta['local_path'].apply(os.path.basename)
        file_to_sha = latent_meta.set_index('join_key')['sha256'].to_dict()

        sim_df['sha256'] = sim_df['raw_filename'].map(file_to_sha)
        n_before = len(sim_df)
        sim_df = sim_df.dropna(subset=['sha256']).reset_index(drop=True)

        # Keep only rows whose latent .pt exists
        sim_df['latent_exists'] = sim_df['sha256'].apply(
            lambda h: os.path.exists(os.path.join(data_dir, f'{h}.npz'))
        )
        sim_df = sim_df[sim_df['latent_exists']].reset_index(drop=True)

        # Remove flipped samples
        sim_df['is_flipped'] = sim_df['Name'].str.contains('_flipped')
        sim_df = sim_df[~sim_df['is_flipped']].reset_index(drop=True)

        # Fix AoA to a single value
        sim_df = sim_df[sim_df['AoA_deg'] == 6].reset_index(drop=True)

        self._stats = {
            'sim_rows_total': n_before,
            'sim_rows_matched': len(sim_df),
            'unique_geometries': sim_df['sha256'].nunique(),
        }

        # ---- Normalization stats ----
        self.target_stats = {}
        if self.normalize_targets:
            for k in self.target_keys:
                vals = sim_df[k].values.astype(np.float64)
                self.target_stats[k] = {
                    'mean': float(np.nanmean(vals)),
                    'std': float(np.nanstd(vals)) + 1e-8,
                }

        # ---- Instance list: (sha256, {target_k: val}) ----
        self.instances = []
        for _, row in sim_df.iterrows():
            targets = {k: float(row[k]) for k in self.target_keys}
            self.instances.append((row['sha256'], targets))

        self._latent_cache = {}

    def split(self, test_ratio: float = 0.2, seed: int = 42) -> Tuple['RewardDataset', 'RewardDataset']:
        """
        Split into train/test datasets by geometry (no leakage).
        Returns (train_dataset, test_dataset).
        """
        rng = np.random.RandomState(seed)
        unique_shas = list(set(sha for sha, _ in self.instances))
        rng.shuffle(unique_shas)
        n_test = max(1, int(len(unique_shas) * test_ratio))
        test_shas = set(unique_shas[:n_test])

        train_ds = copy.copy(self)
        test_ds = copy.copy(self)
        train_ds.instances = [(s, t) for s, t in self.instances if s not in test_shas]
        test_ds.instances = [(s, t) for s, t in self.instances if s in test_shas]
        # Share the cache
        train_ds._latent_cache = self._latent_cache
        test_ds._latent_cache = self._latent_cache
        return train_ds, test_ds

    def __len__(self):
        return len(self.instances)

    def __str__(self):
        lines = [
            self.__class__.__name__,
            f'  - Instances: {len(self)}',
            f'  - Unique geometries: {self._stats["unique_geometries"]}',
            f'  - Filtered: {self._stats["sim_rows_matched"]}/{self._stats["sim_rows_total"]}',
            f'  - Targets: {self.target_keys}',
        ]
        if self.normalize_targets:
            for k, s in self.target_stats.items():
                lines.append(f'    - {k}: mean={s["mean"]:.4f}, std={s["std"]:.4f}')
        return '\n'.join(lines)

    def _load_latent(self, sha256: str):
        if sha256 not in self._latent_cache:
            path = os.path.join(self.data_dir, f'{sha256}.npz')
            with np.load(path) as f:
                coords = torch.tensor(f['coords'], dtype=torch.int32)
                feats = torch.tensor(f['feats'], dtype=torch.float32)
                # Prepend batch column: (x, y, z) → (batch, x, y, z)
                if coords.shape[1] == 3:
                    coords = torch.cat([
                        torch.zeros(coords.shape[0], 1, dtype=coords.dtype),
                        coords
                    ], dim=1)
                self._latent_cache[sha256] = sp.SparseTensor(feats, coords)
        return self._latent_cache[sha256]

    def _normalize_target(self, targets: Dict[str, float]) -> torch.Tensor:
        vals = []
        for k in self.target_keys:
            v = targets[k]
            if self.normalize_targets:
                v = (v - self.target_stats[k]['mean']) / self.target_stats[k]['std']
            vals.append(v)
        return torch.tensor(vals, dtype=torch.float32)

    def get_denormalize_fn(self) -> Callable:
        """Returns a function to map normalized predictions back to physical units."""
        stats = self.target_stats
        keys = self.target_keys
        def denorm(pred: torch.Tensor) -> Dict[str, torch.Tensor]:
            out = {}
            for i, k in enumerate(keys):
                v = pred[..., i]
                if k in stats:
                    v = v * stats[k]['std'] + stats[k]['mean']
                out[k] = v
            return out
        return denorm

    def __getitem__(self, index) -> Dict[str, Any]:
        try:
            sha256, targets = self.instances[index]
            latent = self._load_latent(sha256)
            pack = {
                'scalar_target': self._normalize_target(targets),
            }
            if isinstance(latent, sp.SparseTensor):
                pack['x'] = latent
            elif isinstance(latent, dict):
                pack.update(latent)
            else:
                pack['x'] = latent
            return pack
        except Exception as e:
            print(f'Error loading instance {index}: {e}')
            return self.__getitem__(np.random.randint(0, len(self)))

    @staticmethod
    def collate_fn(batch, split_size=None):
        if split_size is None:
            group_idx = [list(range(len(batch)))]
        else:
            loads = []
            for b in batch:
                if isinstance(b.get('x'), sp.SparseTensor):
                    loads.append(b['x'].feats.shape[0])
                else:
                    loads.append(1)
            group_idx = load_balanced_group_indices(loads, split_size)

        packs = []
        for group in group_idx:
            sub = [batch[i] for i in group]
            pack = {}
            for k in sub[0].keys():
                if isinstance(sub[0][k], sp.SparseTensor):
                    pack[k] = sp.sparse_cat([b[k] for b in sub], dim=0)
                elif isinstance(sub[0][k], torch.Tensor):
                    pack[k] = torch.stack([b[k] for b in sub])
                elif isinstance(sub[0][k], list):
                    pack[k] = sum([b[k] for b in sub], [])
                else:
                    pack[k] = [b[k] for b in sub]
            packs.append(pack)
        return packs[0] if split_size is None else packs