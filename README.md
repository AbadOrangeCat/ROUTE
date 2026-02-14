# ROUTE: Robust Online Unsupervised Triage and Ensemble

A **reliability-first, label-free** way to adapt fake news detectors when the topic suddenly changes.

---

## What this repository is about

Fake news detectors often work well on the topic they were trained on, but fail when the topic changes. This is common in real life:

- Yesterday’s “medical misinformation” data looks different from today’s “COVID-19 misinformation”
- New events introduce new vocabulary, writing styles, and repeating claim patterns
- Labels for the new topic usually arrive late, or never arrive at all

This repo contains a reference implementation of the method described in the paper:

> **ROUTE (Robust Online Unsupervised Triage and Ensemble)**

ROUTE adapts a text classifier to a new domain **without using any labeled examples from the target domain**. Instead of trusting one adaptation algorithm, ROUTE tries a small set of “routes” (safe choices), checks which ones behave reliably on unlabeled target data, and avoids risky updates using a **source validation safety check**.

---

## The main idea in plain language

Before we go into setup and commands, here is the core story in one minute:

1. You have a classifier trained on an old topic (the **source domain**) where labels exist.
2. A new topic appears (the **target domain**), but you do not have labels yet.
3. You still want good performance on the new topic.

ROUTE does this by:

- preparing several candidate ways to adapt the model (the **routes**)
- scoring each route using reliability signals computed from **unlabeled target text**
- rejecting routes that damage performance on the known **source validation set**
- finally deploying either:
  - **one selected route** (when one option clearly looks best), or
  - a small weighted **ensemble** (when multiple options look similarly safe)

---

## Quick glossary (terms used consistently in this repo)

To make the README readable for non-specialists, the following terms are used throughout:

- **Domain**: a setting where texts share a similar topic and writing style (e.g., “general medical” vs “COVID-19”).
- **Source domain**: an older domain where you have labeled data.
- **Target domain**: a new domain where you have unlabeled data at deployment time.
- **Domain adaptation**: updating a model so it works better on the target domain.
- **Unsupervised / label-free adaptation**: adaptation that uses no target labels.
- **Route**: a complete adaptation recipe (example: “do nothing”, “self-train”, “consistency training”).
- **Pseudo-label**: a temporary label produced by the model itself (used carefully, because it can be wrong).
- **Ensemble**: combining multiple models by averaging their predicted probabilities.
- **Safety check**: a rule that prevents destructive updates by verifying the model still performs well on source validation data.

---

## What the code implements from the paper

This repository follows the same high-level pipeline described in the paper:

### 1) Base detector

- A BERT-style text classifier is fine-tuned on labeled **source** data.
- Optionally, the encoder is warmed up on unlabeled **target** text using masked language modeling  
  (often called **Domain-Adaptive PreTraining**, abbreviated as **DAPT**).

### 2) Candidate adaptation routes

The code builds a small pool of candidates, including:

- **NONE**: do not adapt beyond the base model
- **Self-training** (three variants):
  - `st_hard`: use only very confident pseudo-labels
  - `st_balanced`: like `st_hard`, but keeps both classes represented to avoid “class collapse”
  - `st_sched`: start very strict, then gradually relax the confidence threshold
- **FixMatch + Mean Teacher** (`fixmatch_mt`):
  - encourages prediction consistency under text perturbations (token masking)
  - uses an EMA “teacher” model to stabilize pseudo-labels

### 3) Reliability signals (computed without target labels)

Each candidate is evaluated on a target holdout split using signals such as:

- prediction confidence/entropy patterns
- class balance and collapse indicators
- embedding-based prototype agreement (source class “anchors”)
- embedding cluster separation
- stability under dropout (measured by KL divergence across stochastic passes)
- reverse validation (train a lightweight classifier on pseudo-labeled target embeddings and test on source validation)

### 4) Safety checks

A route is rejected if it:

- drops too much on source validation accuracy
- collapses to predicting mostly one class
- fails minimum prototype coverage/agreement thresholds

### 5) Final decision

The selector chooses:

- a single route (conservative choice when one candidate clearly behaves best), or
- an ensemble of top safe candidates (to reduce randomness and improve stability)

---

## Repository layout and expected folders

This section explains how to place your data so the script can find it.

A typical layout looks like this:

```text
repo_root/
  <main_entry_script>.py
  sourcedata/
    source_train.csv
    source_validation.csv
    source_test.csv
  targetdata/
    train.csv
    val.csv
    test.csv
  processed_acl/
    books/
      positive.review
      negative.review   (or negatiev.review in some releases)
      unlabeled.review
    dvd/
      ...
    electronics/
      ...
    kitchen/
      ...
  outputs_routeB_ultra_multidatav2/
    ...
```

**Notes:**

- The code automatically checks and removes exact duplicate overlaps between train/val/test (within each domain) to avoid accidental contamination.
- The ACL Amazon dataset loader is designed to be robust to minor filename variations.

---

## Installation

You only need a standard Python deep learning environment. GPU is recommended but not required.

### Create and activate an environment (example)

```bash
conda create -n route python=3.10 -y
conda activate route
```

### Install core dependencies

```bash
pip install torch transformers numpy pandas
```

If you want exact reproducibility across machines, pin versions in a `requirements.txt` and install from that file.

---

## Data preparation

This repository supports two evaluation styles:

1. A simple CSV-based “source vs target” task (useful for medical → COVID-19 style transfers)
2. The Amazon reviews benchmark (12 cross-category transfers)

The next two subsections explain each format.

---

### 1) CSV task (source and target as CSV files)

You need:

**Source:**

- `sourcedata/source_train.csv`
- `sourcedata/source_validation.csv`
- `sourcedata/source_test.csv`

**Target:**

- `targetdata/train.csv` (unlabeled during adaptation)
- `targetdata/val.csv` (unlabeled holdout for reliability scoring)
- `targetdata/test.csv` (labels optional; used only for final evaluation if present)

Each CSV should contain:

- a **text** column (common names like `text`, `content`, `tweet`, `review` are auto-detected)
- a **label** column for source files (common names like `label`, `y`, `target` are auto-detected)

If your column names differ, the script will guess, but you can also rename columns to make it explicit.

**Important behavior:**

- Target labels are **never** used for adaptation or selection.
- Target test labels are used **only** for final reporting, if available.

---

### 2) Amazon ACL reviews benchmark

The code expects a folder like:

```text
processed_acl/<domain>/
  positive.review
  negative.review (or negatiev.review)
  unlabeled.review
```

Domains supported by default:

- `books`, `dvd`, `electronics`, `kitchen`

How splitting works:

- source labeled (positive/negative) is split into train/val/test with a fixed seed
- target unlabeled is split into:
  - target train (used for adaptation updates)
  - target val (used for reliability scoring)
- target labeled test is built from target positive/negative files

---

## Running the experiments

The code is designed as a “single entrypoint” runner. It will:

- run multiple random seeds
- run multiple tasks (CSV task + ACL tasks)
- write detailed logs and JSON summaries

Before running, open the `Config` dataclass inside the script to confirm:

- `run_main_csv_task` / `run_acl_tasks`
- `acl_root_rel` points to your ACL folder
- `seeds` list is what you want
- your machine has enough memory (BERT + multiple candidates)

Then run:

```bash
python <main_entry_script>.py
```

By default, outputs go to a folder like:

```text
outputs_routeB_ultra_multidatav2/
```

If you want to run fewer tasks (for example, only the CSV task), set:

```python
cfg.run_main_csv_task = True
cfg.run_acl_tasks = False
```

---

## Understanding the outputs

Each task produces both raw logs and summarized results.

A typical task folder contains:

- `all_seed_results.json`  
  Per-seed, per-method results (includes baselines and each candidate).

- `multi_seed_summary.json`  
  Mean/std across seeds for each method (accuracy, macro-F1, balanced accuracy, etc.).

- `seed_metas.json`  
  Per-seed selection decisions and metadata.

Each seed also stores:

- `routeB_ultra_candidate_metrics.json`  
  The reliability signals and safety metrics used by the selector.

- `routeB_ultra_selection.json`  
  The final chosen route and decision reasons.

If target labels exist, the script can also write:

- `selector_metric_rows.csv` and `selector_metric_correlation.csv`  
  Helpful for checking which reliability signals correlate with true target accuracy.

---

## Reported results (from the paper)

To help readers connect code to claims, here are the key numbers reported in the manuscript:

### Medical → COVID-19 fake news transfer

- a strong DAPT-based baseline reaches **52.96%** target accuracy
- ROUTE improves to **69.01%** accuracy and **0.6776** macro-F1  
  (all without labeled target examples)

### Amazon reviews (12 transfers across 4 categories)

- ROUTE improves average accuracy by **0.92** points
- beats “no adaptation” in **10/12** transfers
- reduces variance across random seeds

This repo is built to reproduce that evaluation pattern:  
same backbone, multiple candidate routes, label-free selection, and careful reporting over multiple seeds.

---

## How to extend ROUTE in this codebase

This repository is intentionally modular: it treats “adaptation” as a small set of pluggable routes.

Before adding a new route, remember the main constraint:

> The selector must not use target labels.

### Adding a new route

A good workflow is:

1. Implement an adaptation function similar to:
   - `adapt_selftrain_closed(...)`
   - `adapt_fixmatch_mean_teacher(...)`
2. Add it into the `candidates = [...]` list inside `run_seed(...)`.
3. Make sure you also compute reliability signals for it, so the selector can compare it fairly.

### Adding a new dataset / domain split

For a new dataset, you have two options:

- Use the CSV interface and create:
  - source train/val/test CSVs
  - target train/val/test CSVs

**Or**

- Add a new “task kind” similar to the ACL loader:
  - define a new `TaskSpec.kind`
  - implement `load_<newtask>_task(...)`

---

## Practical notes and limitations

- **Reliability signals are proxies, not guarantees.**  
  They reduce risk, but they cannot perfectly detect every failure mode.

- **Self-training can amplify mistakes.**  
  The whole point of ROUTE is to avoid “always self-train” behavior by checking safety and reliability first.

- **Compute cost is higher than a single method.**  
  ROUTE trains multiple candidates. The pool is small by design, but it is still more expensive than one fixed pipeline.

- **This is research code.**  
  It is useful for experiments, ablations, and reproducibility. Production deployments need additional monitoring, logging, and human oversight.
