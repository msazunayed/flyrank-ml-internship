# Predicting Future Content Performance Decline for Search-Intelligence Review Prioritization

**FlyRank ML Internship Capstone** · Lane: Refresh / Content Opportunity Scoring
**Author:** Shakil Ahmed Zunayed ([@MSAZunayed](https://msazunayed.github.io))

> A decision-support model that ranks web pages by their risk of losing search impressions next month, so a content/SEO reviewer knows which 50 pages to look at first.

**Demo video (unlisted, ~5 min):** `https://youtu.be/wDECsVVbFCo`
**Notebook:** [`work/notebooks/capstone.ipynb`](work/notebooks/capstone.ipynb)

---

## What it does and for whom

A content or SEO team cannot manually inspect every page. This project answers one question:

> Can search and engagement signals measured *before* a decision point predict future content-performance decline, and produce a better review queue than a transparent hand-written rule?

- **For:** content and SEO reviewers with limited time.
- **Input:** one row per pseudonymized client-content page, built from March 2026 search and engagement data.
- **Output:** a ranked review queue. Pages are ordered by predicted probability of decline, and each page gets a plain-language reason code.
- **What it is not:** it does not rewrite, merge, prune, or delete content. It only ranks candidates for a human to review.

## Key results

| Method | Precision@50 | ROC-AUC | Avg. Precision |
|---|---|---|---|
| Week-4 rule baseline (hand-written impression, position, CTR logic) | 0.48 | n/a | n/a |
| **Logistic Regression (final model)** | **0.66** | 0.573 | 0.484 |
| Random Forest | 0.50 | 0.557 | 0.458 |
| Histogram Gradient Boosting | 0.54 | 0.583 | 0.486 |

- Precision@50 rose from **0.48 to 0.66**, a **37.5% relative improvement** (about 24 vs. about 33 proxy-positive pages in the top 50).
- Test set: 15,191 pages from 9 clients never seen in training. Base rate of the decline label is 0.412.
- Logistic Regression is the final model because it gave the best Precision@50, the metric that matches real review capacity. Its simplicity also makes it easy to explain.
- A random-row split gave Precision@50 of 0.72, but the client-grouped split gave 0.66. **0.66 is the number to trust**, because random splits leak client information.

> **Note on reproducibility:** the notebook's audit cell recomputes every model metric and compares it to the reported table. Logistic Regression matches exactly. Random Forest and Gradient Boosting Precision@50 came out lower on recompute (0.50 and 0.54 vs. 0.58 and 0.56 originally reported), so the table above uses the recomputed values. The conclusion is unchanged: Logistic Regression is still the best at the top of the queue.

## How it works (architecture)

```
 FlyRank internship warehouse (Hugging Face, gated, pseudonymized)
                  │  fact_content_daily_performance
                  ▼
        DuckDB reads parquet directly (hf://...)
                  │
     ┌────────────┴─────────────┐
     ▼                          ▼
 March 2026 features      April 2026 outcome
 (impressions, clicks,    (April impressions)
  CTR, avg position,             │
  sessions)                      ▼
     │              future_decline = 1 if April impressions
     │              are at least 20% lower than March
     ▼                          │
     └────────────┬─────────────┘
                  ▼
   Client-grouped train/test split (GroupShuffleSplit)
                  │
      ┌───────────┼───────────────┐
      ▼           ▼               ▼
 Rule baseline  Logistic Reg.   Random Forest /
 (Week 4)       (final)         Gradient Boosting
      └───────────┼───────────────┘
                  ▼
     Precision@50 comparison on same held-out pages
                  ▼
   Ranked top-50 review queue + reason codes (CSV, charts)
```

### Data and label

- **Table used:** `fact_content_daily_performance`, aggregated to one row per client-content page.
- **Feature window:** March 2026. **Outcome window:** April 2026. The windows do not overlap.
- **Filter:** pages need at least 100 Google Search Console impressions in March, so tiny pages don't dominate the percentage change.
- **Final dataset:** 100,893 client-content observations.
- **Features (5):** impressions, clicks, CTR, average search position, sessions.
- **Label:** `future_decline = 1` when April impressions are at least 20% lower than March. This is an operational proxy, not a claim that the page is defective.

### Leakage checks

April impressions, the target, target-derived fields, FlyRank product outputs and scores, and identifiers are **not** used as predictors. Client names, domains, URLs, raw queries, and titles are excluded entirely.

## Setup (reproduce from scratch)

**Prerequisites:** Python 3.11, VS Code with the Jupyter extension, and a Hugging Face account with approved access to the gated dataset `FlyRank/internship-warehouse`.

1. Clone the repo and enter it:
   ```bash
   git clone <YOUR_REPO_URL>
   cd flyrank-ml-internship
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   ```
3. Install the packages:
   ```bash
   pip install duckdb huggingface_hub scikit-learn pandas numpy matplotlib ipykernel
   ```
4. Log in to Hugging Face once so DuckDB can read the gated dataset:
   ```bash
   huggingface-cli login            # newer versions: hf auth login
   ```
   Alternatively, set the `HF_TOKEN` environment variable before starting VS Code.
5. Open `work/notebooks/capstone.ipynb` in VS Code, click **Select Kernel**, and choose the `.venv` Python.
6. Click **Run All**. Charts and tables are saved to `work/outputs/`.

**Offline option:** set the environment variable `FLYRANK_DATA` to a local folder with the same layout as the warehouse. Never commit data files to the repo.

## Usage and outputs

Running the notebook top to bottom walks through nine sections: question, data, methodology, results, model comparison, limitations, ranked recommendations, and a demo outline. It writes these files to `work/outputs/`:

| File | What it shows |
|---|---|
| `capstone_metrics.json` / `capstone_model_metrics.json` / `baseline_metrics.json` | Precision@50, ROC-AUC, average precision, relative improvement |
| `model_vs_baseline_precision50.png` | Model vs. rule baseline chart |
| `capstone_model_comparison.png` | Comparison of the three models |
| `capstone_feature_coefficients.png` | Logistic Regression coefficients per feature |
| `capstone_top20_scores.png` | Top-20 pages by model score |
| `capstone_ranked_recommendations.csv` | The ranked review queue with `model_score` and `reason_code` per page |

**Example reason codes** (interpretable context for reviewers): weak CTR at visible positions, high-visibility pages with relatively weak click capture, and pages where several model signals combine.

## Guardrails

- **Human in the loop.** The output is a ranked queue, never an automatic action. Content should not be auto-rewritten, merged, pruned, or deleted.
- **Privacy.** Only public-safe, aggregated, pseudonymized measurements are used. No client names, URLs, or raw queries appear anywhere in the notebook or outputs.
- **Careful claims.** Results are described as observed, measured, and directional, and are decision support only.
- **Leakage prevention.** Client-grouped holdout with zero client overlap between train and test.

## Limitations

- **One time transition only.** The experiment covers March to April 2026, so stability across seasons and future periods is not established.
- **Proxy label.** A 20% impression drop is not a measure of content quality or of whether a page needs a refresh.
- **Unbalanced data.** Client histories and measurement availability differ across the warehouse.
- **GA4 zero-fill.** Sessions are filled with zero where no value exists, which can mix real zero activity with missing analytics data. This should be fixed in a future iteration.
- **Observational, not causal.** The results do not show that changing a recommended page will recover its search performance, and they do not reveal Google's ranking algorithm or prove any feature is a ranking factor.
- **Modest signal.** ROC-AUC is about 0.57, so the model is useful for top-of-queue ranking but is not a strong general classifier.
- **Not deployment-ready.** Forward-in-time validation across multiple feature and outcome windows is needed first.
- **Before acting on a recommendation,** a reviewer should consider search intent, query mix, seasonality, SERP changes, related-page consolidation, measurement availability, and traffic volume.

## Repository layout

```
flyrank-ml-internship/
├── work/
│   ├── notebooks/
│   │   ├── capstone.ipynb                  # final end-to-end notebook
│   │   ├── w01_research_question.ipynb
│   │   ├── w02_ml_task_framing.ipynb
│   │   ├── w03_data_contract.ipynb
│   │   ├── w03_feature_leakage_check.ipynb
│   │   ├── w04_baseline_score.ipynb
│   │   ├── w05_model.ipynb
│   │   ├── w06_validation_audit.ipynb
│   │   └── w07_action_playbook.ipynb
│   └── outputs/                            # generated charts, metrics, ranked CSV
├── DATA_USE.md
├── LICENSE
└── README.md
```

## Acknowledgments and data credit

Data: pseudonymized FlyRank internship warehouse release. Thanks to [FlyRank](https://flyrank.ai) for the dataset and the internship program. Use of the data is subject to the terms in `DATA_USE.md`.
