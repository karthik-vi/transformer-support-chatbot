"""Intent classifiers: TF-IDF + logistic regression baseline, and a fine-tuned Transformer."""
from __future__ import annotations

import time

import numpy as np


class TfidfLR:
    def fit(self, X, y):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline

        self.pipe = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1),
                                  LogisticRegression(max_iter=3000, C=10.0))
        self.pipe.fit(X, y)
        self.labels = list(self.pipe.classes_)
        return self

    def predict_proba(self, X):
        return self.pipe.predict_proba(X), self.labels


class TransformerClassifier:
    def __init__(self, model_name: str = "distilbert-base-uncased", max_len: int = 64):
        self.model_name, self.max_len = model_name, max_len

    def fit(self, X, y, X_val=None, y_val=None, epochs: int = 3, lr: float = 5e-5, batch_size: int = 32, seed: int = 0, log=print):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        self.labels = sorted(set(y))
        lid = {l: i for i, l in enumerate(self.labels)}
        self.tok = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=len(self.labels), id2label=dict(enumerate(self.labels)), label2id=lid)
        enc = self.tok(list(X), truncation=True, max_length=self.max_len, padding=True, return_tensors="pt")
        yt = torch.tensor([lid[l] for l in y])
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        steps = epochs * ((len(y) + batch_size - 1) // batch_size)
        sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
        self.history = []
        for ep in range(epochs):
            self.model.train()
            order = rng.permutation(len(y))
            t0, tot = time.time(), 0.0
            for b in range(0, len(order), batch_size):
                j = torch.tensor(order[b: b + batch_size])
                out = self.model(input_ids=enc["input_ids"][j], attention_mask=enc["attention_mask"][j], labels=yt[j])
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                sch.step()
                opt.zero_grad()
                tot += out.loss.item() * len(j)
            h = {"epoch": ep, "train_loss": tot / len(y), "epoch_s": round(time.time() - t0, 1)}
            if X_val is not None:
                p, labs = self.predict_proba(X_val)
                h["val_acc"] = float(np.mean(np.array(labs)[p.argmax(1)] == np.array(y_val)))
            self.history.append(h)
            log(h)
        return self

    def predict_proba(self, X, batch_size: int = 128):
        import torch

        self.model.eval()
        outs = []
        with torch.no_grad():
            for b in range(0, len(X), batch_size):
                enc = self.tok(list(X[b: b + batch_size]), truncation=True, max_length=self.max_len, padding=True, return_tensors="pt")
                outs.append(torch.softmax(self.model(**enc).logits, -1).numpy())
        return np.concatenate(outs), self.labels

    def save(self, path):
        self.model.save_pretrained(path)
        self.tok.save_pretrained(path)

    @classmethod
    def load(cls, path):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        obj = cls(str(path))
        obj.tok = AutoTokenizer.from_pretrained(path)
        obj.model = AutoModelForSequenceClassification.from_pretrained(path)
        obj.labels = [obj.model.config.id2label[i] for i in range(obj.model.config.num_labels)]
        return obj
