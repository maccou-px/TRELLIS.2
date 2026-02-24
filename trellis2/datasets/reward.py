from typing import *
import os
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from ..modules import sparse as sp
from ..utils.data_utils import load_balanced_group_indices


class RewardDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        *,
        sim_csv: str,
        latent_meta_csv: str,
        sample_names_csv: str,
        target_keys: List[str],
        join_key: str,
        cond_keys: List[str] = [],
        target_scaler: StandardScaler = None,
        cond_scaler: StandardScaler = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.target_keys = target_keys
        self.cond_keys = cond_keys
        self.join_key = join_key
        self.sim_csv = sim_csv
        self.latent_meta_csv = latent_meta_csv
        self.sample_names_csv = sample_names_csv

        # Load & join
        sim_df = pd.read_csv(sim_csv)
        meta = pd.read_csv(latent_meta_csv)
        sha_map = meta.set_index(meta['local_path'].apply(os.path.basename))['sha256']
        sim_df['sha256'] = sim_df[join_key].map(sha_map)
        sim_df = sim_df.dropna(subset=['sha256'])
        sim_df = sim_df[sim_df['sha256'].apply(
            lambda h: os.path.exists(os.path.join(data_dir, f'{h}.npz'))
        )]

        # Filter by sample names
        if sample_names_csv:
            names = set(pd.read_csv(sample_names_csv, header=None)[0].str.strip())
            sim_df = sim_df[sim_df['Name'].isin(names)]

        sim_df = sim_df.reset_index(drop=True)

        # Build instances
        self.instances = []
        for _, row in sim_df.iterrows():
            t = np.array([float(row[k]) for k in target_keys], dtype=np.float32)
            c = np.array([float(row[k]) for k in cond_keys], dtype=np.float32) if cond_keys else np.empty(0, dtype=np.float32)
            self.instances.append((row['sha256'], t, c))

        # Fit or reuse scalers
        if target_scaler is not None:
            self.target_scaler = target_scaler
        else:
            self.target_scaler = StandardScaler().fit(np.stack([t for _, t, _ in self.instances]))

        if cond_keys:
            if cond_scaler is not None:
                self.cond_scaler = cond_scaler
            else:
                self.cond_scaler = StandardScaler().fit(np.stack([c for _, _, c in self.instances]))
        else:
            self.cond_scaler = None

        self._cache = {}

    def __len__(self):
        return len(self.instances)

    def __str__(self):
        lines = [f'RewardDataset: {len(self)} instances, {len(set(s for s,_,_ in self.instances))} geometries']
        for i, k in enumerate(self.target_keys):
            lines.append(f'  {k}: mean={self.target_scaler.mean_[i]:.4f}, std={self.target_scaler.scale_[i]:.4f}')
        for i, k in enumerate(self.cond_keys):
            if self.cond_scaler:
                lines.append(f'  {k}: mean={self.cond_scaler.mean_[i]:.4f}, std={self.cond_scaler.scale_[i]:.4f}')
        return '\n'.join(lines)

    def get_denormalize_fn(self):
        s = self.target_scaler
        keys = self.target_keys
        def denorm(pred):
            scale = torch.tensor(s.scale_, dtype=pred.dtype, device=pred.device)
            mean = torch.tensor(s.mean_, dtype=pred.dtype, device=pred.device)
            pred = pred * scale + mean
            return {k: pred[..., i] for i, k in enumerate(keys)}
        return denorm

    def _load(self, sha):
        if sha not in self._cache:
            with np.load(os.path.join(self.data_dir, f'{sha}.npz')) as f:
                coords = torch.tensor(f['coords'], dtype=torch.int32)
                feats = torch.tensor(f['feats'], dtype=torch.float32)
            if coords.shape[1] == 3:
                coords = torch.cat([torch.zeros(coords.shape[0], 1, dtype=coords.dtype), coords], 1)
            self._cache[sha] = sp.SparseTensor(feats, coords)
        return self._cache[sha]

    def __getitem__(self, idx):
        sha, targets, conds = self.instances[idx]
        pack = {
            'x': self._load(sha),
            'y': torch.from_numpy(self.target_scaler.transform(targets[None])[0]),
        }
        if self.cond_keys:
            pack['mod'] = torch.from_numpy(self.cond_scaler.transform(conds[None])[0])
        return pack

    @staticmethod
    def collate_fn(batch, split_size=None):
        groups = [list(range(len(batch)))] if split_size is None else \
            load_balanced_group_indices([b['x'].feats.shape[0] for b in batch], split_size)
        packs = []
        for grp in groups:
            sub = [batch[i] for i in grp]
            pack = {}
            for k in sub[0]:
                v0 = sub[0][k]
                if isinstance(v0, sp.SparseTensor):
                    pack[k] = sp.sparse_cat([b[k] for b in sub], dim=0)
                elif isinstance(v0, torch.Tensor):
                    pack[k] = torch.stack([b[k] for b in sub])
                else:
                    pack[k] = [b[k] for b in sub]
            packs.append(pack)
        return packs[0] if split_size is None else packs