# Plan

Last updated 2026-11-19. Time budget: about 8-10 hours a week. Hard date: January 15 (Macalester
Regular Decision). By then a public v0.1 must exist.

## Scope

The contribution is the dataset plus a careful paired evaluation. The uncertainty methods are known
(Hamidieh et al., 2026; Alfarano et al., 2026), so the project does not claim a new method.

Conditions per item:

| Condition | Text | Purpose |
|---|---|---|
| `en` | MMLU-Redux English | baseline |
| `mn_mt` | raw Google Translate | what an unverified translation gives |
| `mn` | human post-edited | the dataset |
| `en_para` | English paraphrase (about 100 items) | separates memorization from language |

Confidence signals (all from letter probabilities, one model run per prompt):
own max probability, ensemble-averaged probability, leave-one-out agreement (how much the other
models support this model's answer). Verbalized confidence: at most one model, as a reference point.

## Priorities

| Priority | Item | Status |
|---|---|---|
| Must | Item selection from MMLU-Redux, filtering, stratified order | done |
| Must | Probability scoring pipeline with rotations and decision-point readout | done, tested on CPU |
| Must | Translation workflow with hidden answer key and edit-rate checks | done, tested |
| Must | Machine translation of all 1,950 segments, frozen once | week 1 |
| Must | Pilot: 50 items, 8 candidate models, model and instruction choice | week 1-2 |
| Must | Blind post-editing, 150+ items by Jan 10, 300 by Jan 24 | weeks 1-9 |
| Must | Hand-check every item all models get wrong | week 10 |
| Must | Public v0.1 (150+ items, dataset card) before Jan 15 | week 7-8 |
| Should | Native-speaker check of the Mongolian prompt wording | week 1 |
| Should | Email a professor at the National University of Mongolia (draft in docs/) | week 1 |
| Should | Predictions committed before the main runs (PREDICTIONS.md) | before week 3 |
| Should | 100 English paraphrases | weeks 8-9 |
| Should | One full-precision (fp16) run to measure the 4-bit effect | week 9 |
| Could | Second person codes a subset of failure categories | week 10-11 |
| Could | Canary string and a small private split | at v1.0 |
| Cut | MedMCQA, letter-only self-consistency, BioMistral, live demo | |

## Timeline

| Week | Dates | Work | Output |
|---|---|---|---|
| 1 | Nov 23-29 | Machine translation in Sheets; check prompt wording; Kaggle and HF accounts; email professor; start batch 1 | `items_mn_mt.jsonl` |
| 2 | Nov 30-Dec 6 | Pilot on Kaggle; fill PREDICTIONS.md; translate about 5 h | final model list |
| 3-4 | Dec 7-20 | English runs (all 390 items); translate (SAT week: translation only) | English results |
| 5-6 | Dec 21-Jan 3 | English analysis; professor reviews a 30-item sample; reach 120+ verified | |
| 7-8 | Jan 4-15 | Reach 150+; publish v0.1 on Hugging Face with dataset card | **v0.1 before Jan 15** |
| 9 | Jan 16-24 | Finish 300; paraphrases; Mongolian runs; fp16 check | all conditions run |
| 10-12 | Jan 25-Feb 14 | Paired analysis, failure coding, report, v1.0 | report, v1.0 |

## Risks

| Risk | What we do |
|---|---|
| Mongolian accuracy near chance (25%) for most models | Pilot rule keeps only models at or above 40%; larger candidates as backup |
| Translation takes longer than 25-40 h | Batches are ordered so any prefix is a fair sample; 150 items is still a valid v0.1 |
| Kaggle setup problems (phone verification, gated models) | Do the account steps in week 1, before the pilot is needed |
| Unanimous wrong answers are answer-key errors | Hand-check each one before calling it a model failure |
| Overclaiming | Say "first human-verified parallel English-Mongolian biology set we could find", cite Bokhbat/LLM_Benchmark and MM-Eval |
