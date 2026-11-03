import numpy as np
from fastapi.testclient import TestClient

from chatbot.bot import SupportBot
from chatbot.data import download, load_split
from chatbot.models import TfidfLR


class Fixed:
    labels = ["card_arrival", "lost_or_stolen_card"]

    def __init__(self, p):
        self.p = p

    def predict_proba(self, X):
        return np.array([self.p]), self.labels


def test_escalates_below_threshold():
    bot = SupportBot(Fixed([0.55, 0.45]), threshold=0.6)
    r = bot.reply("where is my card")
    assert r.action == "escalate" and r.intent == "card_arrival"
    r = SupportBot(Fixed([0.9, 0.1]), threshold=0.6).reply("where is my card")
    assert r.action == "answer" and "card arrival" in r.text


def test_tfidf_baseline_learns_banking77_subset(tmp_path):
    d = download(__import__("pathlib").Path("data/banking77"))
    X, y = load_split(d / "train.csv")
    keep = [i for i, l in enumerate(y) if l in ("card_arrival", "lost_or_stolen_card", "exchange_rate")]
    Xs, ys = [X[i] for i in keep], [y[i] for i in keep]
    m = TfidfLR().fit(Xs[::2], ys[::2])
    p, labels = m.predict_proba(Xs[1::2])
    acc = np.mean(np.array(labels)[p.argmax(1)] == np.array(ys[1::2]))
    assert acc > 0.85


def test_api_health_and_chat(monkeypatch):
    monkeypatch.setenv("MODEL_DIR", "does/not/exist")  # forces the TF-IDF fallback so the test needs no fine-tuned weights
    from chatbot import api

    api._state.clear()
    c = TestClient(api.app)
    h = c.get("/health").json()
    assert h["status"] == "ok" and h["model"] == "tfidf_fallback"
    r = c.post("/chat", json={"message": "I lost my card, what do I do?"}).json()
    assert r["action"] in ("answer", "escalate") and 0 <= r["confidence"] <= 1
