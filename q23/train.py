from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from .core import DATA, OUT, MODEL_ID, MODEL_REVISION, Predictor, block_mask, load_embedding, loss_fn, metrics, pack_split, predict, read_pkl, save_json, seed_all, take


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--baseline", action="store_true", help="Train the same architecture without missing-span augmentation")
    args = parser.parse_args()
    seed_all(args.seed)
    torch.set_num_threads(8)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    raw = read_pkl(DATA / "附件2-数据集特征文件/aligned_50.pkl")
    train, stats = pack_split(raw["train"], device)
    valid, _ = pack_split(raw["valid"], device, stats)
    embedding = load_embedding(device)
    model = Predictor(embedding).to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=5e-4, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=3e-5)
    rng = np.random.default_rng(args.seed)
    best, best_epoch, best_state, history = -1e9, 0, None, []
    started = time.time()
    for epoch in range(1, args.epochs+1):
        model.train()
        order = torch.randperm(len(train["ids"]), device=device)
        losses = []
        for indices in order.split(args.batch_size):
            batch = take(train, indices)
            if not args.baseline:
                batch = block_mask(batch, rng)
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch)
            loss = loss_fn(pred, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.)
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        logits, reg = predict(model, valid)
        measured = metrics(logits, reg, valid)
        # One fixed selection rule; all diagnostic masking studies run after selection.
        score = measured["macro_f1"] - .08 * measured["mae"]
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), **measured})
        if score > best:
            best, best_epoch = score, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if not k.startswith("embedding.")}
        if epoch == 1 or epoch % 5 == 0:
            print(f"epoch={epoch} loss={np.mean(losses):.4f} valid_f1={measured['macro_f1']:.4f} valid_mae={measured['mae']:.4f}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    suffix = "_baseline" if args.baseline else ""
    torch.save({"state_dict": best_state, "epoch": best_epoch, "seed": args.seed,
                "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
                "architecture": "Predictor(hidden=96, dropout=0.2)"}, OUT / f"model{suffix}.pt")
    np.savez(OUT / "normalization.npz", **stats)
    save_json(OUT / f"train_history{suffix}.json", {"settings": vars(args), "device": device,
              "elapsed_seconds": time.time()-started, "best_epoch": best_epoch,
              "history": history})
    print(f"saved best epoch {best_epoch}", flush=True)


if __name__ == "__main__":
    main()
