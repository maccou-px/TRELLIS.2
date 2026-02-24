from typing import *
import os, json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from scipy import stats as scipy_stats
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from ..basic import BasicTrainer
from ...datasets.reward import RewardDataset
from ...utils.data_utils import recursive_to_device


class RewardTrainer(BasicTrainer):

    def __init__(
        self,
        *args,
        eval_splits: Dict[str, str] = {},
        eval_batch_size: int = 256,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.target_keys = self.dataset.target_keys
        self.eval_batch_size = eval_batch_size

        # Its pretty hacky to get multiple splits in. What im doing is initialising the dataset
        # with train data only, and fitting scalars. Then im reinitialising additional eval datasets
        # with the same scalers here. The eval data only gets accessed by run_snapshot
        train_ds = self.dataset

        # Build eval splits reusing train scalers
        self.eval_splits = {'train': train_ds}
        for name, csv in eval_splits.items():
            self.eval_splits[name] = RewardDataset(
                data_dir=train_ds.data_dir,
                sim_csv=train_ds.sim_csv,
                latent_meta_csv=train_ds.latent_meta_csv,
                join_key=train_ds.join_key,
                target_keys=train_ds.target_keys,
                cond_keys=train_ds.cond_keys,
                sample_names_csv=csv,
                target_scaler=train_ds.target_scaler,
                cond_scaler=train_ds.cond_scaler,
            )

        if self.is_master:
            parts = [f'{k}: {len(v)}' for k, v in self.eval_splits.items()]
            print(f'  Splits: {" / ".join(parts)}')

    def training_losses(self, x, y, mod=None, **kwargs):
        pred = self.training_models['model'](x, mod=mod, **kwargs)
        if pred.dim() == 1:
            pred = pred.unsqueeze(-1)
        if y.dim() == 1:
            y = y.unsqueeze(-1)
        mse = F.mse_loss(pred, y)
        with torch.no_grad():
            mae = F.l1_loss(pred, y)
        return {'loss': mse}, {'mae': mae}  # Only mse is backpropagated

    @torch.no_grad()
    def snapshot_dataset(self, *a, **kw):
        pass

    @torch.no_grad()
    def snapshot(self, suffix=None, batch_size=None, **kwargs):
        suffix = suffix or f'step{self.step:07d}'
        if self.is_master: print(f'\nEval ({suffix})...')
        results = self.run_snapshot(suffix=suffix)
        if self.is_master:
            out = os.path.join(self.output_dir, 'samples', suffix)
            os.makedirs(out, exist_ok=True)
            with open(os.path.join(out, 'eval_metrics.json'), 'w') as f:
                json.dump(results, f, indent=2)

    @torch.no_grad()
    def run_snapshot(self, suffix=None, **kwargs):
        all_metrics, all_preds = {}, {}
        for name, ds in self.eval_splits.items():
            p, t = self._predict(ds, name)
            all_metrics[name] = self._metrics(p, t)
            all_preds[name] = (p, t)

        if self.is_master:
            names = list(all_metrics.keys())
            keys = list(all_metrics[names[0]].keys())
            print(f'  {"":>20s} ' + ' / '.join(f'{n:>10s}' for n in names))
            for k in keys:
                print(f'  {k:>20s} ' + ' / '.join(f'{all_metrics[n][k]:>10.4f}' for n in names))

            if suffix:
                out = os.path.join(self.output_dir, 'samples', suffix)
                os.makedirs(out, exist_ok=True)
                denorm = self.dataset.get_denormalize_fn()
                self._plot(all_preds, out, denorm)

        flat = {f'{n}/{k}': v for n, m in all_metrics.items() for k, v in m.items()}
        if self.is_master:
            for k, v in flat.items():
                self.writer.add_scalar(f'eval/{k}', v, self.step)
        return flat

    def _predict(self, ds, desc='eval'):
        loader = DataLoader(ds, batch_size=self.eval_batch_size, shuffle=False, collate_fn=ds.collate_fn)
        preds, targets = [], []
        for data in tqdm(loader, desc=f'Eval {desc}', disable=not self.is_master):
            data = recursive_to_device(data, 'cuda')
            preds.append(self.models['model'](**data).cpu())
            targets.append(data['y'].cpu())
        p, t = torch.cat(preds), torch.cat(targets)
        if p.dim() == 1:
            p = p.unsqueeze(-1)
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        return p, t

    def _metrics(self, preds, targets):
        m = {'mae': F.l1_loss(preds, targets).item(), 'mse': F.mse_loss(preds, targets).item()}
        denorm = self.dataset.get_denormalize_fn()
        p, t = denorm(preds), denorm(targets)
        for i, k in enumerate(self.target_keys):
            if i < preds.shape[-1]:
                m[f'{k}_mae'] = F.l1_loss(p[k], t[k]).item()
                if preds.shape[0] > 1:
                    m[f'{k}_r2'] = self._compute_r2(preds[:, i], targets[:, i])
                    m[f'{k}_rho'] = self._compute_spearman(p[k].numpy(), t[k].numpy())
        return m

    def _compute_r2(self, pred, target):
        return float(torch.corrcoef(torch.stack([pred, target]))[0, 1]) ** 2

    def _compute_spearman(self, pred, target):
        return scipy_stats.spearmanr(target, pred).correlation if len(pred) > 1 else 0

    def _plot(self, all_preds, out_dir, denorm):
        """Create the evaluation figure."""
        splits = list(all_preds.keys())
        n_targets = len(self.target_keys)
        n_splits = len(splits)
        
        fig, axes = plt.subplots(n_targets, n_splits, figsize=(5 * n_splits, 5 * n_targets))
        if n_targets == 1:
            axes = axes.reshape(1, -1)
        if n_splits == 1:
            axes = axes.reshape(-1, 1)
        
        for col_idx, split in enumerate(splits):
            preds, targets = all_preds[split]
            predictions, targets_denorm = denorm(preds), denorm(targets)
            
            for row_idx, key in enumerate(self.target_keys):
                ax = axes[row_idx, col_idx]
                p = predictions[key].numpy()
                t = targets_denorm[key].numpy()

                mae = np.mean(np.abs(p - t))
                r2 = np.corrcoef(p, t)[0, 1] ** 2 if len(p) > 1 else 0
                rho = self._compute_spearman(t, p)

                ax.scatter(t, p, alpha=0.5, s=15, marker='x', color='b')
                lim = [min(t.min(), p.min()), max(t.max(), p.max())]
                ax.plot(lim, lim, 'r--', alpha=0.7, lw=1.5)

                if row_idx == n_targets - 1:
                    ax.set_xlabel('Target', fontsize=10)
                if col_idx == 0:
                    ax.set_ylabel('Prediction', fontsize=10)

                title = f'{split} — {key}\nMAE: {mae:.4f} | ρ: {rho:.4f} | R²: {r2:.4f}'
                ax.set_title(title, fontsize=9)
                ax.set_aspect('equal', adjustable='datalim')

        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, 'all_predictions.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
