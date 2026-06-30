"""
Pure helpers for the representation probe (component 3).

Kept separate from scripts/probe_polysemy.py (which loads the model via the GPT/torch.compile
stack) so they can be imported and unit-tested with only numpy + torch — no model, no
torch._dynamo. Used by the probe runner to fit a linear sense-decoder on hidden states and
bucket its accuracy by left-context length.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch

# Context-length buckets (number of preceding form tokens); a token with context c is
# assigned to the largest edge <= c. Log-spaced so both short- and long-context regimes get
# their own bucket regardless of the model's sequence length.
CTX_EDGES = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512)


def ctx_bucket(c: int) -> int:
    """Largest CTX_EDGES value <= c (the bucket label)."""
    b = CTX_EDGES[0]
    for e in CTX_EDGES:
        if e <= c:
            b = e
        else:
            break
    return b


def fit_linear_probe(X_train, y_train, X_test, num_classes, *, device="cpu",
                     steps=300, lr=0.05, weight_decay=1e-4, seed=0):
    """Fit a multinomial logistic-regression probe (one Linear layer) and return test preds.

    Features are standardized using train statistics. Returns the predicted class per test
    row (numpy int array). Pure torch so it runs anywhere; small enough for CPU at our scales.
    """
    torch.manual_seed(seed)
    Xtr = torch.from_numpy(np.asarray(X_train, dtype=np.float32)).to(device)
    ytr = torch.from_numpy(np.asarray(y_train, dtype=np.int64)).to(device)
    Xte = torch.from_numpy(np.asarray(X_test, dtype=np.float32)).to(device)
    mu = Xtr.mean(0, keepdim=True)
    sd = Xtr.std(0, keepdim=True).clamp_min(1e-6)
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    clf = torch.nn.Linear(Xtr.shape[1], num_classes).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=lr, weight_decay=weight_decay)
    lossf = torch.nn.CrossEntropyLoss()
    for _ in range(steps):
        opt.zero_grad()
        loss = lossf(clf(Xtr), ytr)
        loss.backward()
        opt.step()
    with torch.no_grad():
        preds = clf(Xte).argmax(1).cpu().numpy()
    return preds


def bucket_accuracy(preds, labels, ctx_lens) -> Tuple[Dict[int, Tuple[float, int]], float]:
    """Accuracy per context bucket: {bucket_edge: (acc, n)} plus overall accuracy."""
    preds = np.asarray(preds); labels = np.asarray(labels); ctx_lens = np.asarray(ctx_lens)
    correct = (preds == labels)
    by_bucket: Dict[int, list] = {}
    for c, ok in zip(ctx_lens, correct):
        agg = by_bucket.setdefault(ctx_bucket(int(c)), [0, 0])
        agg[0] += int(ok); agg[1] += 1
    acc = {b: (n_ok / n, n) for b, (n_ok, n) in sorted(by_bucket.items())}
    overall = float(correct.mean()) if correct.size else 0.0
    return acc, overall
