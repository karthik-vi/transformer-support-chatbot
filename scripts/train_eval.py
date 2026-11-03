"""Fine-tune a Transformer intent classifier on Banking77 and evaluate the support-bot policy on the held-out test set.

Splits: official train (10,003) -> 90% fit / 10% validation (stratified, seed); official test (3,080) held out.
The escalation threshold is picked on VALIDATION (largest auto-answer rate with wrong-answer rate <= --max-wrong),
then applied once to test.

  python scripts/train_eval.py --out results/metrics.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from chatbot.data import download, load_split  # noqa: E402
from chatbot.models import TfidfLR, TransformerClassifier  # noqa: E402


def policy_curve(probs, labels, y_true):
    conf = probs.max(1)
    pred = np.array(labels)[probs.argmax(1)]
    ok = pred == np.array(y_true)
    rows = []
    for t in np.round(np.linspace(0.0, 0.99, 100), 2):
        ans = conf >= t
        rows.append({"threshold": float(t), "auto_answer_rate": float(ans.mean()),
                     "auto_resolved_correct_rate": float((ans & ok).mean()),
                     "wrong_answer_rate": float((ans & ~ok).mean()), "escalation_rate": float((~ans).mean()),
                     "precision_when_answering": float(ok[ans].mean()) if ans.any() else None})
    return rows


def pick_threshold(curve, max_wrong):
    ok = [r for r in curve if r["wrong_answer_rate"] <= max_wrong]
    return max(ok, key=lambda r: r["auto_resolved_correct_rate"])["threshold"] if ok else 0.99


def summarize(name, probs, labels, y, thr):
    pred = np.array(labels)[probs.argmax(1)]
    at = next(r for r in policy_curve(probs, labels, y) if abs(r["threshold"] - thr) < 1e-9)
    return {"model": name, "accuracy": round(accuracy_score(y, pred), 4), "macro_f1": round(f1_score(y, pred, average="macro"), 4),
            "threshold": thr, **{k: round(v, 4) if isinstance(v, float) else v for k, v in at.items() if k != "threshold"}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-wrong", type=float, default=0.02, help="max share of ALL queries answered wrongly")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("results/metrics.json"))
    ap.add_argument("--save", type=Path, default=Path("artifacts/model"))
    ap.add_argument("--limit-train", type=int, default=0, help="debug: subsample training set")
    a = ap.parse_args()
    d = download(Path("data/banking77"))
    X, y = load_split(d / "train.csv")
    Xte, yte = load_split(d / "test.csv")
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.1, stratify=y, random_state=a.seed)
    if a.limit_train:
        Xtr, ytr = Xtr[: a.limit_train], ytr[: a.limit_train]
    rep = {"data": "Banking77 (PolyAI, CC BY 4.0)", "n_fit": len(Xtr), "n_val": len(Xva), "n_test": len(Xte), "seed": a.seed}

    t0 = time.time()
    base = TfidfLR().fit(Xtr, ytr)
    pv, lv = base.predict_proba(Xva)
    thr_b = pick_threshold(policy_curve(pv, lv, yva), a.max_wrong)
    pt, lt = base.predict_proba(Xte)
    rep["tfidf_logreg"] = {**summarize("tfidf_logreg", pt, lt, yte, thr_b), "train_s": round(time.time() - t0, 1)}
    print(rep["tfidf_logreg"], flush=True)

    t0 = time.time()
    tr = TransformerClassifier(a.model).fit(Xtr, ytr, Xva, yva, epochs=a.epochs, seed=a.seed)
    rep["transformer_train_s"] = round(time.time() - t0, 1)
    rep["transformer_history"] = tr.history
    pv, lv = tr.predict_proba(Xva)
    curve_val = policy_curve(pv, lv, yva)
    thr = pick_threshold(curve_val, a.max_wrong)
    t0 = time.time()
    pt, lt = tr.predict_proba(Xte)
    rep["transformer_test_infer_ms_per_query"] = round(1000 * (time.time() - t0) / len(Xte), 2)
    rep["transformer"] = summarize(a.model, pt, lt, yte, thr)
    rep["transformer_policy_curve_test"] = policy_curve(pt, lt, yte)[::5]
    # most confused intent pairs on test
    pred = np.array(lt)[pt.argmax(1)]
    from collections import Counter
    rep["top_confusions_test"] = [{"true": t, "pred": p, "n": n} for (t, p), n in
                                  Counter((t, p) for t, p in zip(yte, pred) if t != p).most_common(10)]
    print(rep["transformer"], flush=True)
    a.save.mkdir(parents=True, exist_ok=True)
    tr.save(a.save)
    (a.save / "policy.json").write_text(json.dumps({"threshold": thr, "max_wrong": a.max_wrong}))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
