"""Support bot = intent classifier + resolution templates + confidence-based escalation to a human."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

RESPONSES = Path(__file__).resolve().parents[2] / "configs" / "responses.yaml"


@dataclass
class Reply:
    intent: str
    confidence: float
    action: str  # "answer" | "escalate"
    text: str


class SupportBot:
    def __init__(self, classifier, threshold: float, responses_path: Path = RESPONSES):
        self.clf = classifier  # must expose predict_proba(texts) -> (probs, labels)
        self.threshold = threshold
        cfg = yaml.safe_load(Path(responses_path).read_text())
        self.templates = cfg["intents"]
        self.default = cfg["default"]
        self.escalation = cfg["escalation"]

    def reply(self, message: str) -> Reply:
        probs, labels = self.clf.predict_proba([message])
        j = int(probs[0].argmax())
        intent, conf = labels[j], float(probs[0][j])
        if conf < self.threshold:
            return Reply(intent, conf, "escalate", self.escalation)
        return Reply(intent, conf, "answer", self.templates.get(intent, self.default).strip())
