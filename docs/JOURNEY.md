# The journey: 0.10926 → 0.108467 in seven stages

This is the narrative behind the notebooks: what each stage tried, what the results said, and how that shaped the next
step. Scores are public-leaderboard error rates; "wrong rows" is that score × 10,086.

## Starting point

An earlier CatBoost model with frequency encoding and Optuna tuning scored **0.10926** (1,102 wrong rows, rank 372).
A review of the older scripts found several validation problems worth avoiding: target encoding fitted before
cross-validation, early stopping on the scoring fold, in-sample stacking scores and unseeded tuning. Every stage below
uses leak-free, repeated stratified K-fold CV with the same splits for every model.

The data: 23,524 training and 10,086 test respondents, 14.1% positive. The rate varies a lot by country (Kenya 25%,
Rwanda 11.5%, Tanzania 9%, Uganda 9%), and `year` maps one-to-one onto country, so it was dropped. There is no ID
leak: test IDs form a block after the training IDs and neighbouring labels are uncorrelated.

## Stage 1 — a clean baseline ensemble · 1,101 wrong rows

CatBoost on GPU, LightGBM with native categoricals and a logistic regression on one-hot *pairwise crosses* of the
binned features. Blended with weights that minimise out-of-fold log loss. Result: **0.109161**, a small step past the
starting point and the first move up the leaderboard.

## Stage 2 — more diversity, and a key observation · 1,097 wrong rows

Added CatBoost with exact age and household size as categories plus target-free count features, a PyTorch
entity-embedding network, three-way crosses for the logistic regression, and a per-country logistic stacker.

The probability blend scored **0.108765** and the stacker 0.109459. Lining up the three submissions so far revealed
that **CV log loss ranked them exactly in leaderboard order, while CV error rate did not**. Error rate moves in steps
of one row (≈0.00004), so it is too noisy to choose between close models. From here on, every decision used held-out
log loss.

## Stage 3 — ordered boosting and competing combiners · 1,101 wrong rows

CatBoost on CPU defaults to *ordered boosting* on small data, a different algorithm from the GPU's plain boosting. It
became the best single model and was three times faster. Four combiners competed on held-out log loss; the plain
probability blend won again.

The submission scored worse (1,101 rows) despite a slightly better CV, which led to the most useful discovery of the
project: **every leaderboard score is an exact multiple of 1/10,086**, so the leaderboard counts wrong rows on the
full test set. Stage 3 changed only 22 rows versus stage 2, and stage 2 happened to win 13 of them. Reshuffling the
same models moves a few borderline rows at random.

## Stage 4 — stronger trees, full-data refits · 1,100 wrong rows

Four CPU CatBoost variants (base, extra features, slow learning rate, deeper trees with 3-way category combinations),
test predictions that mix fold models with models refit on 100% of the rows, and SLSQP blend weights. Best CV so far
(0.27729), but the leaderboard stayed within noise. The deep CatBoost took 47% of the blend: tree models were
saturating.

## Stage 5 — new model families · 1,094 wrong rows ★

If better trees no longer help, try models that learn differently. Research into the current state of the art
(the [TabArena](https://arxiv.org/abs/2506.16791) benchmark) pointed to tabular foundation models and modern MLPs:

- **TabICL**: a transformer pretrained on synthetic tables; it reads the training rows as context and predicts in one
  forward pass, with no training loop.
- **TabM** and **RealMLP** from `pytabkit`, with their tuned defaults.
- **CatBoost tuned by Optuna**: 94 seeded, time-boxed trials with pruning, starting from stage 4's best settings; the
  final model was scored on fold splits the search never saw.

The new families improved the blend by −0.00039 log loss, the largest genuine gain of the project, and the leaderboard
followed: **0.108467, rank 247 → 187**. TabICL and TabM took the largest blend weights; RealMLP was slow and weak.

## Stage 6 — scaling what worked · 1,099 wrong rows

More TabICL estimators, TabICL on an ordinal encoding and eight TabM seeds. CV did not move (+0.00004): the bigger
TabICL mostly replaced the old one instead of adding information. The model zoo had plateaued.

## Stage 7 — TabPFN, and a lesson about CV · 1,105 wrong rows

TabPFN v3.5 was the strongest single model ever seen in CV (0.27607 log loss) and took 84% of the blend, giving the
best CV of the project (0.27622). On the leaderboard it was among the worst submissions: of the 49 rows that differed
from stage 5, stage 5 was right on 30.

A plausible explanation is **pretraining contamination**. Recent foundation models are partly trained on real public
tables, and this dataset has circulated publicly for years. If the training rows (with labels) were in TabPFN's
pretraining data, its out-of-fold predictions on them would be optimistic, while the test labels were never public.
It is unproven, but CV and leaderboard agreed in every earlier stage and disagreed sharply here.

## Where the ceiling is

- Fifteen models from five families, tuned and untuned, all land between 11.1% and 11.4% CV error.
- 32% of training rows have an exact feature-duplicate, and 7% of those disagree on the label: no model can get those
  right.
- Stages 1–7 moved the leaderboard by 7 rows in total; the top public scores (≈0.105, ≈1,060 rows, mostly from 2019)
  are about 35 rows better than anything reached here with these 10 columns and no external data.

## What I'd take to the next competition

1. Find out the leaderboard's resolution and noise before reading anything into a score.
2. Choose by the CV metric that tracks the leaderboard, and check that it keeps tracking it.
3. Build a reusable OOF/test prediction store early; it made stages 5–7 cheap.
4. Once trees saturate, add genuinely different model families instead of tuning harder.
5. Treat a foundation model's CV with suspicion on public datasets.
