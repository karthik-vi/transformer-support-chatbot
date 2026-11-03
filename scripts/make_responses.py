"""Generate configs/responses.yaml: one PLACEHOLDER resolution text per Banking77 intent.
These are generic, invented texts. A real deployment would have a content team write the resolution flows."""
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from chatbot.data import download, load_split  # noqa: E402

d = download(Path("data/banking77"))
_, y = load_split(d / "train.csv")
intents = {}
for it in sorted(set(y)):
    readable = it.replace("_", " ").rstrip("?")
    intents[it] = (f"It looks like your question is about: {readable}. "
                   f"[PLACEHOLDER resolution flow for '{it}' - link the help-centre article / in-app action here.]")
Path("configs/responses.yaml").write_text(yaml.safe_dump({
    "default": "Thanks - I have noted your question. [PLACEHOLDER default answer]",
    "escalation": "I want to make sure you get the right answer, so I'm passing this to a support agent.",
    "intents": intents}, sort_keys=True, width=200))
print(len(intents), "intents written")
