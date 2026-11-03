# Data provenance

| Item | Value |
|---|---|
| Dataset | Banking77 (Casanueva et al., "Efficient Intent Detection with Dual Sentence Encoders", NLP4ConvAI 2020) |
| Source | https://github.com/PolyAI-LDN/task-specific-datasets (`banking_data/train.csv`, `test.csv`), downloaded 2026-11-03; also mirrored at https://huggingface.co/datasets/PolyAI/banking77 |
| License | CC BY 4.0 (Hugging Face dataset card `license: cc-by-4.0`, checked 2026-11-03) |
| Size | 13,083 customer queries, 77 intents: official train 10,003, official test 3,080 |
| How obtained | `chatbot.data.download` (called by every script) |
| Splits used | official train → 90% fit / 10% validation (stratified, seed 0); official test held out and used once per reported model |

**Not Jocata data.** Banking77 is a public set of online-banking queries in English. The resolution texts in
`configs/responses.yaml` are generated **placeholders** (`scripts/make_responses.py`), not real bank policies.

## Limitations

- Single-turn queries only; no multi-turn dialogue, no account context, no actual resolution outcomes.
  "Resolution" here = the bot answers automatically **and** picks the correct intent. Whether the customer's problem
  was solved by that answer is not measurable with this dataset.
- Some intents overlap heavily (e.g. `card_arrival` vs `card_delivery_estimate`); see top confusions in EXPERIMENTS.
- English only.
