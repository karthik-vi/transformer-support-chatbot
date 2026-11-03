"""FastAPI demo: POST /chat {"message": "..."} -> intent, confidence, answer-or-escalate.  GET /health.

  MODEL_DIR=artifacts/model uvicorn chatbot.api:app --port 8000
If MODEL_DIR has no fine-tuned model, a TF-IDF baseline is trained on startup so the demo still runs (marked in /health).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

from .bot import SupportBot

app = FastAPI(title="support-chatbot (Banking77 reconstruction)")
_state: dict = {}


class ChatIn(BaseModel):
    message: str


def _load():
    if "bot" in _state:
        return _state["bot"]
    md = Path(os.getenv("MODEL_DIR", "artifacts/model"))
    if (md / "config.json").exists():
        from .models import TransformerClassifier

        clf = TransformerClassifier.load(md)
        thr = json.loads((md / "policy.json").read_text())["threshold"] if (md / "policy.json").exists() else 0.5
        _state["model"] = f"transformer:{md}"
    else:
        from .data import download, load_split
        from .models import TfidfLR

        d = download(Path(os.getenv("DATA_DIR", "data/banking77")))
        X, y = load_split(d / "train.csv")
        clf, thr = TfidfLR().fit(X, y), 0.5
        _state["model"] = "tfidf_fallback"
    _state["bot"] = SupportBot(clf, thr)
    return _state["bot"]


@app.get("/health")
def health():
    _load()
    return {"status": "ok", "model": _state["model"], "threshold": _state["bot"].threshold}


@app.post("/chat")
def chat(body: ChatIn):
    r = _load().reply(body.message)
    return {"intent": r.intent, "confidence": round(r.confidence, 4), "action": r.action, "text": r.text}
