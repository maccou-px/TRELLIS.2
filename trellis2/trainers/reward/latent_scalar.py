from typing import *
import os
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from scipy import stats as scipy_stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from easydict import EasyDict as edict

from ..basic import BasicTrainer
from ...modules import sparse as sp
from ...utils.data_utils import recursive_to_device

class RewardTrainer(BasicTrainer):
    """
    Trainer for predicting scalar targets from pre-extracted latents.

    Args:
        models (dict[str, nn.Module]): Models to train.
            Expected keys:
                - 'model': Maps latent → scalar predictions.

        loss_type (str): 'mse', 'l1', 'huber'.
        lambda_reg (float): L2 penalty on predictions.
        target_keys (list[str]): Names of targets (for per-target logging).
    """

    def __init__(
        self,
        *args,
        target_keys: List[str] = ['Cd', 'Cl'],
        test_split: float = 0.2,
        test_split_seed: int = 42,
        max_eval_samples: int = 256,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.target_keys = target_keys
        self.max_eval_samples = max_eval_samples

        if test_split > 0:
            self.train_dataset, self.test_dataset = self.dataset.split(test_split, test_split_seed)
            self.dataset = self.train_dataset
            self.prepare_dataloader()
            if self.is_master:
                print(f'  Train/test split: {len(self.train_dataset)}/{len(self.test_dataset)} instances')
        else:
            self.test_dataset = self.dataset

    def training_losses(self, x, scalar_target: torch.Tensor, **kwargs) -> Tuple[Dict, Dict]:
        pred = self.training_models['model'](x, **kwargs)
        if pred.dim() == 1:
            pred = pred.unsqueeze(-1)
        if scalar_target.dim() == 1:
            scalar_target = scalar_target.unsqueeze(-1)

        mse = F.mse_loss(pred, scalar_target)
        terms = edict(loss=mse, mse=mse)

        with torch.no_grad():
            status = edict(mae=F.l1_loss(pred, scalar_target))

        return terms, status

    # ---- Evaluation ----

    @torch.no_grad()
    def snapshot_dataset(self, *args, **kwargs):
        pass

    @torch.no_grad()
    def snapshot(self, suffix=None, num_samples=None, batch_size=16, verbose=False):
        if self.is_master:
            suffix = suffix or f'step{self.step:07d}'
            print(f'\nEval ({suffix})...')

        results = self.run_snapshot(num_samples, batch_size=batch_size, suffix=suffix)

        if self.is_master:
            import json as _json
            out_dir = os.path.join(self.output_dir, 'samples', suffix)
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, 'eval_metrics.json'), 'w') as f:
                _json.dump(results, f, indent=2)

    @torch.no_grad()
    def run_snapshot(self, num_samples, batch_size, suffix=None, **kwargs) -> Dict:
        split_metrics = {}
        split_preds = {}
        for name, ds in [('train', self.dataset), ('test', self.test_dataset)]:
            n = min(self.max_eval_samples, len(ds))
            preds, targets = self._predict(ds, n, batch_size, desc=name)
            split_metrics[name] = self._compute_metrics(preds, targets)
            split_preds[name] = (preds, targets)

        # Print as "metric: train / test"
        if self.is_master:
            for k in split_metrics['train'].keys():
                tv = split_metrics['train'].get(k, float('nan'))
                ev = split_metrics['test'].get(k, float('nan'))
                print(f'  {k}: {tv:.6f} / {ev:.6f}')

        # Save scatter plots
        if self.is_master and suffix is not None:
            out_dir = os.path.join(self.output_dir, 'samples', suffix)
            os.makedirs(out_dir, exist_ok=True)
            denorm = self.dataset.get_denormalize_fn() if hasattr(self.dataset, 'get_denormalize_fn') else None
            for name, (preds, targets) in split_preds.items():
                self._save_scatter_plots(preds, targets, out_dir, name, denorm)

        # Flatten for logging
        metrics = {}
        for name, m in split_metrics.items():
            for k, v in m.items():
                metrics[f'{name}/{k}'] = v
        return metrics

    def _save_scatter_plots(self, preds, targets, out_dir, split_name, denorm=None):
        """Save pred vs target scatter plot for each target key."""
        if denorm:
            preds_phys = denorm(preds)
            targets_phys = denorm(targets)
        else:
            preds_phys = {k: preds[:, i] for i, k in enumerate(self.target_keys)}
            targets_phys = {k: targets[:, i] for i, k in enumerate(self.target_keys)}

        for key in self.target_keys:
            p = preds_phys[key].numpy()
            t = targets_phys[key].numpy()

            mae = np.mean(np.abs(p - t))
            r2 = float(np.corrcoef(p, t)[0, 1] ** 2) if len(p) > 1 else 0.0
            spearman = float(scipy_stats.spearmanr(t, p).correlation) if len(p) > 1 else 0.0

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(t, p, alpha=0.5, s=15, marker='x', color='#4AABDE')

            # y = x line
            lo = min(t.min(), p.min())
            hi = max(t.max(), p.max())
            margin = (hi - lo) * 0.05
            ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
                    'r--', alpha=0.7, linewidth=1.5, label='y = x')

            ax.set_xlabel(f'{key} (target)')
            ax.set_ylabel(f'{key} (predicted)')
            ax.set_title(
                f'{split_name} — {key}\n'
                f'MAE: {mae:.4f} | Spearman: {spearman:.4f} | R²: {r2:.4f}'
            )
            ax.legend(loc='upper left')
            ax.set_aspect('equal', adjustable='datalim')
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f'{split_name}_{key}.png'), dpi=150)
            plt.close(fig)

    def _predict(self, dataset, num_samples, batch_size, desc='eval'):
        loader = DataLoader(
            copy.deepcopy(dataset),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=dataset.collate_fn if hasattr(dataset, 'collate_fn') else None,
        )
        all_preds, all_targets = [], []
        for i in range(0, num_samples, batch_size):
            batch = min(batch_size, num_samples - i)
            data = next(iter(loader))
            data = {k: v[:batch] for k, v in data.items()}
            data = recursive_to_device(data, 'cuda')
            all_preds.append(self.models['model'](data['x']).cpu())
            all_targets.append(data['scalar_target'].cpu())

        preds = torch.cat(all_preds, dim=0)
        targets = torch.cat(all_targets, dim=0)
        if preds.dim() == 1: preds = preds.unsqueeze(-1)
        if targets.dim() == 1: targets = targets.unsqueeze(-1)
        return preds, targets

    def _compute_metrics(self, preds, targets) -> Dict:
        metrics = {
            'mae': F.l1_loss(preds, targets).item(),
            'mse': F.mse_loss(preds, targets).item(),
        }
        # Per-target metrics
        denorm = self.dataset.get_denormalize_fn() if hasattr(self.dataset, 'get_denormalize_fn') else None
        if denorm:
            p, t = denorm(preds), denorm(targets)
        for i, key in enumerate(self.target_keys):
            if i >= preds.shape[-1]:
                break
            if denorm:
                metrics[f'{key}_mae'] = F.l1_loss(p[key], t[key]).item()
            if preds.shape[0] > 1:
                r = torch.corrcoef(torch.stack([preds[:, i], targets[:, i]]))[0, 1].item()
                metrics[f'{key}_r2'] = r ** 2
        return metrics