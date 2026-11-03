# transformer-support-chatbot

Fine-tuned Transformer support bot: intent classification on customer banking queries, an approved resolution template per intent, and
confidence-based escalation to a human agent, served by FastAPI.

## What this is / is not

| | |
|---|---|
| **Reconstructs** | Resume (Jocata, Machine Learning Engineer, Jan 2024 – Jun 2025), verbatim: *"Improved support resolution rate by 40% by fine-tuning a Transformer-based NLP chatbot; …"* (the AWS half of that bullet is in `aws-ml-platform-template`). |
| **Is** | A reconstruction on the public **Banking77** dataset (CC BY 4.0). Resolution texts in `configs/responses.yaml` are generated placeholders. All metrics measured here. |
| **Is not** | Jocata's chatbot, data, intents or answers. The 40% figure is not reproduced (see Results). |

### Period freeze
Jocata, **cutoff Jun 2025**. Python 3.11; `distilbert-base-uncased` (2019); transformers 4.51.3 (2025-04-14); torch 2.7.0 CPU; scikit-learn 1.6.1;
FastAPI 0.115.12. Lock compiled with `--exclude-newer 2025-06-30`. See `docs/DECISIONS.md`.

## Architecture
```
query ─► DistilBERT (fine-tuned, 77 intents) ─► max softmax ≥ threshold? ──yes──► reviewed template for that intent
                                                        └──no──► escalate to a human agent
threshold chosen on validation: max correct auto-answers s.t. wrong auto-answers ≤ 2% of all queries
FastAPI: POST /chat {message} → {intent, confidence, action, text};  GET /health
```

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
pytest -q                                                          # no fine-tuned weights needed
python scripts/train_eval.py --out results/metrics.json             # ~1.5 h on 4 CPUs (3 epochs); saves artifacts/model
MODEL_DIR=artifacts/model PYTHONPATH=src uvicorn chatbot.api:app --port 8000
curl -s -X POST localhost:8000/chat -H 'content-type: application/json' -d '{"message":"my card still has not arrived"}'
```

## Results (Banking77 official test set, 3,080 queries)

| Model | Accuracy | Correctly auto-resolved | Wrong auto-answers | Escalated |
|---|---|---|---|---|
| TF-IDF + logistic regression | 0.891 | 0.699 | 0.019 | 0.282 |
| **DistilBERT fine-tuned** | **0.899** | **0.746** | **0.017** | **0.237** |

> Resume claim: +40% support resolution rate | Reproduced here: 74.6% vs 69.9% correctly auto-resolved (+4.7 pts, +6.7% relative) against a TF-IDF baseline at ≤2% wrong answers | Gap / why: proxy metric on public single-turn queries; the original baseline and resolution definition are unknown.

## Limitations
Single-turn queries; "resolution" = correct intent + automatic answer, not a solved customer problem; placeholder answers; one seed; CPU-only training
limited the run to 3 epochs.

## How to extend
Multi-turn context; out-of-scope intent detection; calibration (temperature scaling) to widen the auto-answer zone; human-in-the-loop feedback to retrain.
