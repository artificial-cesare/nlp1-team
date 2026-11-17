"""Models that read text or token ids directly (step 6.4 and 6.5), with the same CV protocol as eval_harness.

- `TinyTransformer`: trained from scratch on token ids (2 layers, pre-LayerNorm, masked mean pooling, linear softmax).
  Vocabulary = RoBERTa byte-level BPE ids seen at least `min_freq` times in the training fold; others -> [UNK].
- `FinetuneClassifier`: a pretrained encoder (default LUSR) + mean pooling + linear head, all weights trained
  (the setup of Sun et al. 2025), or only the top `train_layers` layers + head; `head_dropout` before the head.

Protocol: for each outer fold, train on the inner-train part, pick the best epoch on the grouped inner validation
set (early stopping), and score that checkpoint once on the outer fold. By default there is no refit, so each fold
costs one training run. With `refit=True` (fine-tuning only), the model is then retrained from scratch on the whole
outer training fold (inner-train + inner-validation) for the selected number of epochs, and that model is scored,
as eval_harness does for the MLPs. Results come back in the run_cv format, so eval_harness.compare and
shared.analysis work unchanged.
"""

from __future__ import annotations

import copy
import itertools
import math
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score

from shared import eval_harness as H
from shared.tokenization import tokenize

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Search spaces (kept small; logged in the notebook)
TRANSFORMER_GRID = {"d_model": [128, 256], "lr": [5e-4]}
TRANSFORMER_FIXED = {"layers": 2, "heads": 4, "dropout": 0.1, "weight_decay": 0.01, "batch_size": 32,
                     "max_epochs": 60, "patience": 8, "min_freq": 2, "max_len": 512, "tokenizer": "roberta-bpe"}
FINETUNE_FIXED = {"model_id": "rrivera1849/LUSR", "lr": 2e-5, "head_lr": 1e-3, "weight_decay": 0.01, "epochs": 5,
                  "train_layers": None, "head_dropout": 0.0,  # None = train all 12 layers; k = freeze all but the top k  # 10 in the final run
                  "batch_size": 16, "warmup": 0.1, "max_len": 512}


def _grid(grid):
    return [dict(zip(grid, v)) for v in itertools.product(*grid.values())]


# --------------------------------------------------------------------------- from-scratch transformer
class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, n_classes, d_model, layers, heads, dropout, max_len):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model, heads, 4 * d_model, dropout, batch_first=True, norm_first=True,
                                           activation="gelu")
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, ids):
        mask = ids != 0
        h = self.tok(ids) + self.pos(torch.arange(ids.shape[1], device=ids.device))[None]
        h = self.norm(self.enc(h, src_key_padding_mask=~mask))
        m = mask.unsqueeze(-1).float()
        return self.head((h * m).sum(1) / m.sum(1))


def _pad(seqs, max_len):
    L = min(max(len(s) for s in seqs), max_len)
    out = torch.zeros(len(seqs), L, dtype=torch.long)
    for i, s in enumerate(seqs):
        s = s[:L]
        out[i, : len(s)] = torch.as_tensor(s)
    return out


class TransformerClassifier:
    def __init__(self, **p):
        self.p = {**TRANSFORMER_FIXED, **p}

    def _encode(self, docs):
        return [[self.vocab_.get(t, 1) for t in d] or [1] for d in docs]  # 0 = pad, 1 = [UNK]

    def _predict_logits(self, model, docs):
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(docs), 128):
                out.append(model(_pad(docs[i : i + 128], self.p["max_len"]).to(DEVICE)).float().cpu())
        return torch.cat(out)

    def fit(self, docs_tr, y_tr, docs_va, y_va, n_classes, seed=H.SEED):
        p = self.p
        torch.manual_seed(seed)
        g = torch.Generator().manual_seed(seed)
        counts = Counter(t for d in docs_tr for t in d)
        kept = sorted(t for t, c in counts.items() if c >= p["min_freq"])  # vocabulary from the training part only
        self.vocab_ = {t: i + 2 for i, t in enumerate(kept)}
        tr, va = self._encode(docs_tr), self._encode(docs_va)
        model = TinyTransformer(len(self.vocab_) + 2, n_classes, p["d_model"], p["layers"], p["heads"],
                                p["dropout"], p["max_len"]).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
        lossf, yt = nn.CrossEntropyLoss(), torch.as_tensor(y_tr)
        best, wait, self.best_epoch_ = -1.0, 0, 0
        for ep in range(1, p["max_epochs"] + 1):
            model.train()
            perm = torch.randperm(len(tr), generator=g).tolist()
            for i in range(0, len(tr), p["batch_size"]):
                b = perm[i : i + p["batch_size"]]
                opt.zero_grad()
                lossf(model(_pad([tr[j] for j in b], p["max_len"]).to(DEVICE)), yt[b].to(DEVICE)).backward()
                opt.step()
            val = balanced_accuracy_score(y_va, self._predict_logits(model, va).argmax(1).numpy())
            if val > best:
                best, wait, self.best_epoch_, state = val, 0, ep, copy.deepcopy(model.state_dict())
            else:
                wait += 1
                if wait >= p["patience"]:
                    break
        model.load_state_dict(state)
        self.model_, self.val_score_ = model, best
        return self

    def predict_proba(self, docs):
        return torch.softmax(self._predict_logits(self.model_, self._encode(docs)), 1).numpy()


# --------------------------------------------------------------------------- fine-tuned pretrained encoder
class _EncoderHead(nn.Module):
    def __init__(self, model_id, n_classes, train_layers=None, head_dropout=0.0):
        super().__init__()
        from transformers import AutoModel

        self.enc = AutoModel.from_pretrained(model_id)
        self.drop = nn.Dropout(head_dropout)
        self.head = nn.Linear(self.enc.config.hidden_size, n_classes)
        if train_layers is not None:  # freeze embeddings and all but the top `train_layers` transformer layers
            for prm in self.enc.parameters():
                prm.requires_grad = False
            n = len(self.enc.encoder.layer)
            self.frozen = [self.enc.embeddings, *self.enc.encoder.layer[: n - train_layers]]
            for layer in self.enc.encoder.layer[n - train_layers:]:
                for prm in layer.parameters():
                    prm.requires_grad = True
        else:
            self.frozen = []

    def train(self, mode=True):
        super().train(mode)
        for m in self.frozen:  # frozen parts stay in eval mode: no dropout (MPS attention cannot use it without grad)
            m.eval()
        return self

    def forward(self, input_ids, attention_mask):
        h = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).float()
        return self.head(self.drop((h * m).sum(1) / m.sum(1)))  # mean pooling, as for the frozen LUSR embedding


class FinetuneClassifier:
    def __init__(self, **p):
        self.p = {**FINETUNE_FIXED, **p}

    def _batches(self, texts, bs):
        for i in range(0, len(texts), bs):
            enc = self.tok_(list(texts[i : i + bs]), padding=True, truncation=True, max_length=self.p["max_len"],
                            return_tensors="pt")
            yield i, enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE)

    def _logits(self, model, texts):
        model.eval()
        out = []
        with torch.no_grad():
            for _, ids, am in self._batches(texts, 64):
                out.append(model(ids, am).float().cpu())
        return torch.cat(out)

    def fit(self, texts_tr, y_tr, texts_va, y_va, n_classes, seed=H.SEED, stop_after=None):
        """Train for p["epochs"] epochs and keep the best epoch on (texts_va, y_va).

        Refit mode (texts_va=None, stop_after=e): no validation; follow the same p["epochs"]-epoch LR schedule but
        stop after epoch e and keep that model, so the refit sees the LR trajectory that was validated.
        """
        from transformers import AutoTokenizer, get_linear_schedule_with_warmup

        p = self.p
        refit = texts_va is None
        if refit and not stop_after:
            raise ValueError("refit mode (texts_va=None) needs stop_after = number of epochs")
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        self.tok_ = AutoTokenizer.from_pretrained(p["model_id"])
        model = _EncoderHead(p["model_id"], n_classes, p["train_layers"], p["head_dropout"]).to(DEVICE)
        self.n_trainable_ = sum(q.numel() for q in model.parameters() if q.requires_grad)
        opt = torch.optim.AdamW([{"params": [q for q in model.enc.parameters() if q.requires_grad], "lr": p["lr"]},
                                 {"params": model.head.parameters(), "lr": p["head_lr"]}],
                                weight_decay=p["weight_decay"])
        steps = p["epochs"] * math.ceil(len(texts_tr) / p["batch_size"])
        sched = get_linear_schedule_with_warmup(opt, int(p["warmup"] * steps), steps)
        lossf, yt = nn.CrossEntropyLoss(), torch.as_tensor(y_tr)
        texts_tr = np.asarray(texts_tr, dtype=object)
        best, self.best_epoch_, self.history_ = -1.0, 0, []
        for ep in range(1, p["epochs"] + 1):
            model.train()
            perm = rng.permutation(len(texts_tr))
            losses = []
            for i, ids, am in self._batches(texts_tr[perm], p["batch_size"]):
                opt.zero_grad()
                loss = lossf(model(ids, am), yt[perm[i : i + p["batch_size"]]].to(DEVICE))
                loss.backward()
                losses.append(loss.item())
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
            if refit:
                self.history_.append({"epoch": ep, "train_loss": float(np.mean(losses))})
                if ep == stop_after:
                    break
                continue
            val_logits = self._logits(model, list(texts_va))
            val = balanced_accuracy_score(y_va, val_logits.argmax(1).numpy())
            val_loss = lossf(val_logits, torch.as_tensor(y_va)).item()
            self.history_.append({"epoch": ep, "train_loss": float(np.mean(losses)), "val_loss": val_loss,
                                  "val_bal_acc": val})
            if val > best:
                best, self.best_epoch_ = val, ep
                state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        if refit:
            best, self.best_epoch_ = float("nan"), stop_after
        else:
            model.load_state_dict(state)
        self.model_, self.val_score_ = model, best
        return self

    def predict_proba(self, texts):
        return torch.softmax(self._logits(self.model_, list(texts)), 1).numpy()

    def release(self):
        del self.model_
        if DEVICE == "mps":
            torch.mps.empty_cache()


# --------------------------------------------------------------------------- CV loop (run_cv format)
def _refit_finetune(est, p, fixed, texts, y, idx, n_classes, seed):
    """Retrain the selected fine-tuning setup on all of `idx` for est.best_epoch_ epochs (frees `est` first)."""
    epochs, inner_val, history = est.best_epoch_, est.val_score_, est.history_
    est.release()
    est = FinetuneClassifier(**{**fixed, **p}).fit([texts[i] for i in idx], y[idx], None, None, n_classes,
                                                   seed=seed, stop_after=epochs)
    est.val_score_, est.history_ = inner_val, history  # report the selection run's validation curve
    return est


def run_cv_text(texts, y, groups, folds, kind: str, grid: dict | None = None, name: str | None = None,
                verbose: bool = True, refit: bool = False, **fixed) -> dict:
    """kind = "transformer" (token ids from shared.tokenization) or "finetune" (raw text).

    For each outer fold: every grid point is trained on the inner-train part with early stopping on the inner
    validation part; the best one (by inner balanced accuracy) is scored on the outer fold. With refit=True
    (finetune only), the selected setup is first retrained on the whole outer training fold for the selected
    number of epochs.
    """
    if refit and kind != "finetune":
        raise ValueError("refit is implemented for kind='finetune' only")
    texts, y, groups = list(texts), np.asarray(y), np.asarray(groups)
    n_classes = int(y.max()) + 1
    if kind == "transformer":
        data = tokenize(texts, {**TRANSFORMER_FIXED, **fixed}["tokenizer"])
        grid, make = _grid(grid or TRANSFORMER_GRID), lambda p: TransformerClassifier(**fixed, **p)
    else:
        data = texts
        grid = _grid(grid or {"lr": [fixed.get("lr", FINETUNE_FIXED["lr"])]})
        make = lambda p: FinetuneClassifier(**{**fixed, **p})
    rows, oof_pred, oof_proba, t0 = [], np.full(len(y), -1), np.zeros((len(y), n_classes)), time.time()
    for r, k, tr, te in folds:
        H.set_seed(H.SEED + r)
        itr, iva = H.inner_split(tr, groups, seed=H.SEED + r)
        best = None
        for p in grid:
            est = make(p).fit([data[i] for i in itr], y[itr], [data[i] for i in iva], y[iva], n_classes, seed=H.SEED + r)
            if best is None or est.val_score_ > best[0].val_score_:
                if best is not None and hasattr(best[0], "release"):
                    best[0].release()
                best = (est, p)
            elif hasattr(est, "release"):
                est.release()
        est, p = best
        if refit:
            est = _refit_finetune(est, p, fixed, data, y, tr, n_classes, seed=H.SEED + r)
        proba = est.predict_proba([data[i] for i in te])
        pred = proba.argmax(1)
        rows.append({"repeat": r, "fold": k, **H.scores(y[te], pred), "inner_val": est.val_score_, "params": p,
                     "epochs": est.best_epoch_, "n_test": len(te), "n_total": len(tr) + len(te),
                     "history": getattr(est, "history_", None)})
        if r == 0:
            oof_pred[te], oof_proba[te] = pred, proba
        if hasattr(est, "release"):
            est.release()
        if verbose:
            print(f"  fold {r}.{k}: bal.acc {rows[-1]['balanced_accuracy']:.4f} (epoch {est.best_epoch_}, {p}) "
                  f"{time.time() - t0:.0f}s")
    res = {"name": name or kind, "model": kind, "folds": rows, "oof_pred": oof_pred, "oof_proba": oof_proba,
           "seconds": time.time() - t0, "runs": len({f[0] for f in folds}), "fixed": fixed, "refit": refit}
    if verbose:
        ba = H.fold_scores(res)
        print(f"{res['name']:28s} bal.acc {ba.mean():.4f} ± {ba.std():.4f}  ({len(rows)} folds, {res['seconds']:.0f}s)")
    return res


def final_test_text(texts_train, y, groups, texts_test, y_test, kind: str, params: dict | None = None,
                    refit: bool = False, **fixed) -> dict:
    """The single held-out evaluation for a text model: train on an inner grouped split of the full train split
    (early stopping on its validation part), then score the test split once. With refit=True (finetune only), the
    model is then retrained on the whole train split for the selected number of epochs before scoring."""
    if refit and kind != "finetune":
        raise ValueError("refit is implemented for kind='finetune' only")
    H.set_seed(H.SEED)
    y, groups = np.asarray(y), np.asarray(groups)
    itr, iva = H.inner_split(np.arange(len(y)), groups)
    if kind == "transformer":
        tok = {**TRANSFORMER_FIXED, **fixed}["tokenizer"]
        tr_data, te_data = tokenize(list(texts_train), tok), tokenize(list(texts_test), tok)
        est = TransformerClassifier(**fixed, **(params or {"d_model": 256, "lr": 5e-4}))
    else:
        tr_data, te_data = list(texts_train), list(texts_test)
        est = FinetuneClassifier(**fixed, **(params or {}))
    est.fit([tr_data[i] for i in itr], y[itr], [tr_data[i] for i in iva], y[iva], int(y.max()) + 1, seed=H.SEED)
    if refit:
        est = _refit_finetune(est, params or {}, fixed, tr_data, y, np.arange(len(y)), int(y.max()) + 1, seed=H.SEED)
    pred = est.predict_proba(te_data).argmax(1)
    out = {"model": kind, **H.scores(y_test, pred), "inner_val": est.val_score_, "epochs": est.best_epoch_, "pred": pred}
    if hasattr(est, "release"):
        est.release()
    return out
