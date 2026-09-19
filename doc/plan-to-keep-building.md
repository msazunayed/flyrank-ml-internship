# The Plan to Keep Building

## Where the Next Case Study Goes

New case studies live in `docs/case-studies/` in this repository
(`flyrank-ml-internship`), and are indexed from a **Case Studies**
section in the root `README.md`.

### Steps to Add a New Case Study

1. Create a new file: `docs/case-studies/0X-<slug>.md` (next sequential
   number after the last one added).
2. Write it in the three-beat shape:
   - **Problem** — what question or gap motivated the work.
   - **What I did** — the approach, data, and methods used.
   - **What came of it** — the result, what it showed, and any honest
     limitations.
3. Add one line linking the new file from the **Case Studies** section of
   `README.md`.
4. Commit and push. No additional setup is required — the repo structure,
   `requirements.txt`, and Colab access pattern already established in the
   capstone carry over directly.

## Next Real Piece of Work

**Computer vision project (client name pseudonymized, referred to as
"Client XYZ") — image classification and pose estimation.**

- **Task:** Build a model that classifies images into defined categories
  and estimates pose (key points / orientation) for a target subject in
  each image.
- **Why this is next:** The capstone worked with tabular search and
  engagement data. This project extends the portfolio into computer
  vision, a different data modality and a different class of ML problem
  (classification + spatial/keypoint estimation instead of ranking), which
  broadens the range of work demonstrated.
- **Status:** Early-stage — task and problem framing are set; dataset
  scoping, labeling approach, and model architecture are still to be
  decided.

## Keeping the Build Context

The FlyRank Capstone Claude Project is kept active, and a reminder has been
added inside it to resume work on the computer vision project the next time
the project is opened. This preserves voice, stack conventions, and prior
context, so starting the next case study is a short continuation rather
than a rebuild from scratch.
