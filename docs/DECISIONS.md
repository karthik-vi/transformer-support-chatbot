# Architecture Decision Records

## Period freeze

| | |
|---|---|
| Work period reconstructed | Jocata, ML Engineer, Jan 2024 – Jun 2025 |
| Cutoff | Jun 2025 |
| Python | 3.11 |
| Model | `distilbert-base-uncased` (Sanh et al., arXiv 1910.01108, Oct 2019; HF repo `distilbert/distilbert-base-uncased`, createdAt 2022-03-02 (HF migration date), Apache-2.0) |
| Libraries | transformers 4.51.3 (PyPI 2025-04-14), torch 2.7.0 CPU (2025-04-23), scikit-learn 1.6.1 (2025-01-10), FastAPI 0.115.12 (2025-03-23), uvicorn 0.34.2 (2025-04-19), pydantic 2.11.4 (2025-04-29), PyYAML 6.0.2 (2024-08-06), NumPy 2.2.5 (2025-04-19), httpx 0.28.1 (2024-12-06) |

All dates checked on PyPI 2026-11-03; all before the Jun 2025 cutoff. No exceptions.

## ADR-001 Data: Banking77 — 2026-11-03
- **Context.** Need a public support dataset with labelled intents in a financial-services domain.
- **Options.** Banking77 (CC BY 4.0, 77 intents); Bitext customer-support set (synthetic, generated responses); CLINC150 (multi-domain).
- **Decision.** Banking77: real user phrasings, fintech domain, standard held-out test set.
- **Would change my mind.** A public dataset with resolution outcomes (was the issue solved?), which Banking77 lacks.

## ADR-002 Intent classification + templated resolution, not free-text generation — 2026-11-03
- **Context.** In regulated financial support, wrong free-text answers are costly; answers must be reviewable.
- **Options.** (a) fine-tune a seq2seq model to generate answers; (b) fine-tune an encoder for intent, answer from approved templates, escalate when unsure.
- **Decision.** (b). Every automatic answer is a reviewed template; the model only decides which one, or hands off.
- **Tradeoff.** Cannot answer questions outside the 77 intents; those should escalate (confidence threshold).

## ADR-003 Model: DistilBERT fine-tuning on CPU — 2026-11-03
- 66M parameters, 3 epochs, max length 64, AdamW 5e-5, linear warmup/decay, batch 32. Fits CPU training in this environment.
- Baseline: TF-IDF (1–2-grams) + logistic regression, to show what the Transformer adds.

## ADR-004 Evaluation and escalation policy — 2026-11-03
- Classification: accuracy and macro-F1 on the official test set.
- Policy: answer only if max softmax ≥ threshold. Threshold chosen on **validation** as the one maximising correct auto-answers
  subject to wrong auto-answers ≤ 2% of all queries; then applied unchanged to test.
- Reported: auto-resolved-correct rate, wrong-answer rate, escalation rate. These are "resolution-style" proxies on held-out
  queries, not live resolution rates.

## ADR-005 Deployment — 2026-11-03
- FastAPI app with `/chat` and `/health`. If no fine-tuned weights are present it trains the TF-IDF baseline at startup and says so in `/health`.
- Not production-ready: no auth, no conversation state, no PII redaction, no monitoring of confidence drift.
