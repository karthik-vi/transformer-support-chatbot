# Experiment log (newest first)

Banking77: fit 9,002 / val 1,001 (stratified from official train, seed 0) / official test 3,080. CPU, 2026-11-03. Raw: `results/metrics.json`, `results/train_log.txt`.

## EXP-002 — DistilBERT fine-tune vs TF-IDF baseline, with escalation policy (test)

| Model | Accuracy | Macro-F1 | Threshold (from val) | Auto-resolved correctly | Wrong auto-answers | Escalated |
|---|---|---|---|---|---|---|
| TF-IDF (1–2-gram) + LR | 0.891 | 0.891 | 0.58 | 0.699 | 0.019 | 0.282 |
| **DistilBERT, 3 epochs** | **0.899** | **0.898** | 0.68 | **0.746** | **0.017** | **0.237** |

- Policy rule: pick the threshold on validation that maximises correct auto-answers with wrong auto-answers ≤ 2% of all queries; apply unchanged to test.
- DistilBERT: +0.8 accuracy points over TF-IDF but +4.7 points of correctly auto-resolved queries (better-calibrated confidence → fewer escalations at the same error budget).
- Test policy curve (DistilBERT): threshold 0 → 89.9% answered correctly / 10.1% wrong; 0.5 → 83.0% / 4.4%; 0.75 → 67.9% / 1.0%; 0.9 → 32.0% / 0.06%.
- Validation accuracy by epoch: 0.783, 0.883, 0.895. Train time 5,646 s on a shared 4-vCPU box (~27–36 min per epoch); inference 26 ms/query CPU (batched).
- Top test confusions: virtual_card_not_working → get_disposable_virtual_card (15), why_verify_identity → verify_my_identity (12),
  fiat_currency_support → exchange_via_app (7), pending_transfer → transfer_timing (7). Most errors are between near-duplicate intents.
- **Surprise:** the TF-IDF baseline is strong (89.1%) on this single-turn intent task; the Transformer's main value is in the escalation trade-off, not raw accuracy.

## EXP-001 — Smoke tests
Bot policy, TF-IDF baseline on 3 intents, and the FastAPI endpoints (TF-IDF fallback) pass (`tests/test_bot.py`).

## Resume-claim comparison
> Resume claim: improved support resolution rate by 40% | Reproduced here on Banking77 held-out test (n=3,080): correctly auto-resolved share 74.6% with DistilBERT vs 69.9% with a TF-IDF baseline at ≤2% wrong answers (+4.7 points, +6.7% relative) | Gap / why: "resolution rate" here is a proxy (correct intent + automatic answer) on single-turn public queries; the original baseline (likely no bot or a rule-based bot), traffic and resolution definition are unknown.
