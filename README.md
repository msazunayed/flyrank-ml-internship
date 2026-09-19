# Which Pages Should a Reviewer Open First? Ranking Content by Predicted Impression Decline

## FlyRank Machine Learning Internship Capstone

**Lane:** Refresh / Content Opportunity Scoring
**Author:** Shakil Ahmed Zunayed

---

## Overview

This project builds a machine-learning decision-support system that helps content and SEO teams decide which pages should be reviewed first when review capacity is limited.

The system uses historical search and engagement signals to estimate the probability that a content page will experience future performance decline.

The predicted probabilities are then used to create a ranked review queue for human decision-making.

The system does **not** automatically edit, delete, merge, or refresh content.

---

## Problem

A content team may manage thousands of pages but only have enough time to manually review a small number of them.

The main research question is:

> Can observable search and engagement signals measured before a decision point help predict future content-performance decline and produce a more useful ranked review queue than a transparent hand-written rule?

The goal is therefore **prioritization**, not automatic content modification.

---

## Who This Is For

This project is relevant to:

- SEO teams
- content strategists
- digital marketing teams
- search-intelligence analysts
- data scientists working on content-performance monitoring

The system helps a reviewer answer:

> Which pages should I investigate first?

rather than:

> Which pages should automatically be changed?

---

## Data

The project uses the pseudonymized FlyRank internship data warehouse.

The main experiment uses:

- **March 2026** as the feature window
- **April 2026** as the future outcome window
- one pseudonymized client-content page as the unit of analysis
- approximately **100,893 modeling observations**

The five main features are:

- impressions
- clicks
- click-through rate (CTR)
- average search position
- sessions

The prediction target represents future performance decline based on the following time period.

No client names, domains, private URLs, search queries, or personally identifying information are published.

---

## System Architecture

```text
FlyRank Internship Warehouse
            |
            v
      DuckDB Aggregation
            |
      +-----+------+
      |            |
      v            v
March Features   April Outcome
      |            |
      +-----+------+
            |
            v
      Modeling Dataset
            |
            v
   Client-Grouped Holdout
            |
     +------+------+ 
     |             |
     v             v
Rule Baseline    ML Models
                 |
          +------+------+
          |      |      |
          v      v      v
      Logistic   RF   Gradient
      Regression      Boosting
          |
          v
      Evaluation
      Precision@50
          |
          v
   Ranked Review Queue
          |
          v
 Reason Codes + Human Review
```

---

## Machine Learning Approach

The main supervised model is **Logistic Regression**.

I also compared it with:

- Random Forest
- Gradient Boosting
- a transparent rule-based baseline

Logistic Regression was selected as the final model because the operational goal is not simply maximizing general classification accuracy.

The important question is:

> Of the first 50 pages recommended for review, how many are genuinely relevant according to the future-decline target?

For this reason, **Precision@50** is the primary evaluation metric.

Logistic Regression also provides a relatively simple and interpretable modeling approach.

---

## Validation Design

A major design decision in this project was to avoid relying only on a random row split.

Pages belonging to the same client may share similar patterns.

If pages from the same client appear in both training and testing data, performance may look better than it would when the system encounters a completely new client.

Therefore, the final evaluation uses a **client-grouped holdout**.

The grouped split contained:

- **34 training clients**
- **9 test clients**
- **0 client overlap**

This means the model was evaluated on clients it had not seen during training.

The workflow is also time-aware:

```text
March 2026 signals
        |
        v
Model prediction
        |
        v
April 2026 future outcome
```

This helps reduce future-information leakage.

---

## Evaluation Results

The final model comparison used the same operational metric:

**Precision@50**

| Method | Precision@50 |
|---|---:|
| Rule-Based Baseline | 0.48 |
| Logistic Regression | **0.66** |
| Random Forest | 0.58 |
| Gradient Boosting | 0.56 |

### Main Result

The transparent rule-based baseline achieved:

```text
Precision@50 = 0.48
```

Logistic Regression achieved:

```text
Precision@50 = 0.66
```

This corresponds to a relative improvement of approximately:

```text
37.5%
```

In practical terms:

```text
Baseline:
approximately 24 useful candidates
among the first 50 recommendations.

Logistic Regression:
approximately 33 useful candidates
among the first 50 recommendations.
```

The machine-learning ranking therefore improved the usefulness of the limited review queue in this experiment.

---

## Random Split vs Client-Grouped Validation

An earlier random-row evaluation produced:

```text
Precision@50 = 0.72
```

However, the more realistic client-grouped evaluation produced:

```text
Precision@50 = 0.66
```

I report **0.66 as the main result** because the client-grouped holdout is the more trustworthy test.

It measures whether the model can generalize to clients that were completely absent from training.

This difference also demonstrates why validation design matters in machine learning.

---

## Example Workflow

The complete workflow is:

```text
Historical search + engagement data
              |
              v
        Feature creation
              |
              v
     Future-decline target
              |
              v
      Train ML models
              |
              v
     Client-group validation
              |
              v
       Model probability
              |
              v
      Rank candidate pages
              |
              v
        Reason codes
              |
              v
        Human review
```

---

## Ranked Recommendations

The final model converts prediction probabilities into a ranked review queue.

An output can contain fields such as:

```text
rank
content_hash_id
model_score
reason_code
action
```

Example reason codes include:

```text
low_ctr_at_visible_position
page_one_click_gap
high_visibility_low_clicks
```

The purpose of these reason codes is to help a human reviewer understand why a page has been prioritized.

The system recommends **review**, not automatic modification.

---

## Setup

### Option 1 — Google Colab

The easiest way to reproduce the capstone is through Google Colab.

Open:

```text
work/notebooks/capstone.ipynb
```

The repository contains a Colab link for this notebook.

Access to the approved FlyRank internship warehouse is required.

Add the Hugging Face access token to Google Colab Secrets using:

```text
HF_TOKEN
```

Then run:

```text
Runtime → Run all
```

The notebook installs the required libraries and runs the experiment.

---

## Option 2 — Local Setup

Clone the repository:

```bash
git clone https://github.com/Malang43/flyrank-ml-internship-Malang43.git
```

Enter the repository:

```bash
cd flyrank-ml-internship-Malang43
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The starter/reference pipeline can be executed with:

```bash
python scripts/run_all.py
```

The final capstone analysis is located at:

```text
work/notebooks/capstone.ipynb
```

---

## Repository Structure

```text
flyrank-ml-internship-Malang43/
│
├── README.md
├── SETUP.md
├── GUIDE.md
├── DATA_USE.md
├── requirements.txt
│
├── notebooks/
│   └── introductory notebooks
│
├── scripts/
│   └── reference ML pipeline
│
├── work/
│   └── notebooks/
│       ├── w01_research_question.ipynb
│       ├── w02_ml_task_framing.ipynb
│       ├── w03_data_contract.ipynb
│       ├── w03_feature_leakage_check.ipynb
│       ├── w04_signal_audit.ipynb
│       ├── w04_baseline_score.ipynb
│       ├── w05_model.ipynb
│       ├── w06_validation_audit.ipynb
│       ├── w07_action_playbook.ipynb
│       └── capstone.ipynb
│
├── outputs/
│
└── docs/
    └── plan-to-keep-building.md
```

---

## Limitations

This project has several important limitations.

### 1. Limited Time Window

The main experiment evaluates one March-to-April 2026 transition.

Performance cannot automatically be assumed to remain identical across different seasons or future periods.

### 2. Target Is a Proxy

The future-decline target is a measurable proxy for performance decline.

It is not a definitive measure of overall content quality.

### 3. Prediction Is Not Causation

A page receiving a high decline-risk score does **not** prove that editing or refreshing that page will improve its performance.

Demonstrating that a particular intervention caused recovery would require an experiment or another valid causal design.

### 4. Analytics Availability

Engagement measurements may not be equally available across all clients.

Missing session information can sometimes be difficult to distinguish from genuine zero activity.

### 5. Generalization

Additional forward-in-time validation across multiple periods would be required before considering operational deployment.

---

## Important Interpretation

This project should be interpreted as:

```text
decision support
```

and not as:

```text
a prediction of Google's ranking algorithm
```

The model identifies pages that may deserve human investigation first.

It does not claim to know Google's ranking rules and does not guarantee that changing a recommended page will improve traffic.

---

## Case Studies

Additional case studies extending this work are tracked here as they're added.

- [The Plan to Keep Building](docs/plan-to-keep-building.md) — next case study: a computer vision project (image classification + pose estimation) for a pseudonymized client.

To add a new case study, see the steps in `docs/plan-to-keep-building.md`.

---

## AI Transparency

AI tools, including ChatGPT and Claude, were used as development assistants during this project.

AI assistance was used for:

- interpreting assignment requirements
- discussing modeling approaches
- helping draft and refine code
- debugging implementation problems
- reviewing methodology
- improving explanations
- improving documentation
- trimming the demo video and drafting supporting planning documents (e.g. `docs/plan-to-keep-building.md`)

I personally ran the notebooks, inspected the outputs, checked the validation design, reviewed the generated code, made the final modeling decisions, and verified the results reported in this repository.

AI was used as a **development and reasoning assistant**, not as a substitute for testing and verification.

---

## Data Safety

All public results use pseudonymized or aggregated information.

This repository does not intentionally publish:

- client names
- private domains
- private client URLs
- raw private search queries
- credentials
- identifying client information

Results are described using careful terms such as:

- observed
- measured
- directional
- decision-support

rather than claiming that the analysis proves how Google's ranking algorithm works.

---

## Future Improvements

A future version of the project could:

- evaluate several additional future time windows
- test stability across different periods
- improve handling of unavailable analytics measurements
- evaluate probability calibration
- test alternative review-capacity thresholds
- monitor model performance over time
- improve model explanation methods
- evaluate the usefulness of recommendations after human review

The long-term goal is a reliable **human-in-the-loop content review prioritization system**, rather than an automatic content-editing system.

---

## Final Takeaway

The main result of this capstone is:

```text
Rule Baseline Precision@50:       0.48
Logistic Regression Precision@50: 0.66
Relative Improvement:             37.5%
```

The project demonstrates how a transparent machine-learning workflow can improve the prioritization of pages for human review while maintaining careful validation, data safety, interpretability, and honest limitations.

---

## Author

**Shakil Ahmed Zunayed**

FlyRank AI Internship  
Machine Learning Track  
Applied Search Intelligence
