"""Faithful held-out VAL LOSS for cosmos_touch policies.
Instantiates the exact training model+objective (via config), loads a ckpt through
dit_path, points the dataloader at a held-out shard, and averages training_step loss
over a FIXED-SEED set of batches (same noise/sigma across ckpts -> comparable).
Run: torchrun --nproc_per_node=1 -m ... (needs distributed for parallel_state).
"""
import argparse, importlib, os
import numpy as np
import torch
from imaginaire.lazy_config import instantiate
from imaginaire.utils.config_helper import get_config_module, override


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("opts", nargs=__import__("argparse").REMAINDER, default=[])
    ap.add_argument("--nbatch", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()

    cm = get_config_module(a.config)
    config = importlib.import_module(cm).make_config()
    opts = list(a.opts)
    if not opts or opts[0] != "--":
        opts = ["--"] + opts            # override() requires the leading "--" token
    config = override(config, opts)
    config.validate()
    config.freeze()
    trainer = config.trainer.type(config)          # sets up distributed / parallel_state
    model = instantiate(config.model)
    if hasattr(model, "cuda"):
        model = model.cuda()
    model.eval()
    from torch.utils.data import DataLoader
    ds = instantiate(config.dataloader_train.dataset)   # dataset only; skip the custom (train) sampler
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    torch.manual_seed(a.seed)
    losses = []
    it = iter(dl)
    with torch.no_grad():
        for i in range(a.nbatch):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(dl); batch = next(it)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model.training_step(batch, 0)
            losses.append(float(loss))
    losses = np.array(losses)
    print(f"VAL_LOSS mean={losses.mean():.5f} std={losses.std():.5f} n={len(losses)}", flush=True)


if __name__ == "__main__":
    main()
