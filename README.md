# Titanic Disaster – Survival Prediction (Python & R in Docker)

This repository predicts which passengers survived the Titanic disaster (April 15, 1912) using **logistic regression**, implemented twice:

- **Python** (pandas + scikit-learn) in one Docker container
- **R** (readr + dplyr + base `glm`) in a second Docker container

Each script loads the data, prints every cleaning / feature-engineering step to the terminal, trains the model, reports accuracy, and saves predictions for the test set.

> The dataset is **not** included in this repository. Follow Step 2 below to download it.

---

## Repository structure

```
titanic-disaster/
├── .github/
│   └── CODEOWNERS
├── .gitignore
├── README.md
└── src/
    ├── data/                  # put train.csv, test.csv, gender_submission.csv here (git-ignored)
    ├── python/
    │   ├── Dockerfile
    │   ├── requirements.txt
    │   └── titanic_model.py
    └── r/
        ├── Dockerfile
        ├── install_packages.R
        └── titanic_model.R
```

---

## Prerequisites

- [Git](https://git-scm.com/downloads)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) — **make sure it is running** before Step 3
- A free [Kaggle](https://www.kaggle.com/) account (to download the data)

You do **not** need Python or R installed locally; everything runs inside Docker.

---

## Step 1 – Clone the repository

```bash
git clone https://github.com/hyejinlee95/titanic_disaster.git
cd titanic_disaster
```

All commands below are run from this top-level `titanic_disaster` folder.

## Step 2 – Download the data

1. Go to the Kaggle Titanic competition data page: https://www.kaggle.com/competitions/titanic/data
2. Sign in, click **Join Competition** / accept the rules (required before download).
3. Click **Download All**. This gives you `titanic.zip`.
4. Unzip it and move the three CSV files into `src/data/`:

```
src/data/train.csv
src/data/test.csv
src/data/gender_submission.csv
```

**Alternative (Kaggle CLI):** if you have the Kaggle API set up (`pip install kaggle` and an API token in `~/.kaggle/kaggle.json`):

```bash
kaggle competitions download -c titanic -p src/data
unzip src/data/titanic.zip -d src/data
```

Check that the files are in place:

```bash
ls src/data
# gender_submission.csv  test.csv  train.csv
```

## Step 3 – Run the Python model

Build the image:

```bash
docker build -t titanic-python src/python
```

Run the container, mounting the data folder into it:

**macOS / Linux**

```bash
docker run --rm -v "$(pwd)/src/data:/app/data" titanic-python
```

**Windows (PowerShell)**

```powershell
docker run --rm -v "${PWD}/src/data:/app/data" titanic-python
```

**Windows (Command Prompt)**

```cmd
docker run --rm -v "%cd%/src/data:/app/data" titanic-python
```

## Step 4 – Run the R model

Build the image (the first build takes a few minutes while R packages install):

```bash
docker build -t titanic-r src/r
```

Run the container:

**macOS / Linux**

```bash
docker run --rm -v "$(pwd)/src/data:/app/data" titanic-r
```

**Windows (PowerShell)**

```powershell
docker run --rm -v "${PWD}/src/data:/app/data" titanic-r
```

**Windows (Command Prompt)**

```cmd
docker run --rm -v "%cd%/src/data:/app/data" titanic-r
```

---

## What you will see in the terminal

Both scripts print the same seven steps:

| Step | Output |
|------|--------|
| 1. Load training data | Shape of `train.csv` and its first 5 rows |
| 2. Explore | Missing values per column, survival rate overall / by sex / by class |
| 3. Clean & engineer features | One line per change: `[ADD]`, `[FILL]`, `[ENCODE]`, `[DROP]` |
| 4. Fit logistic regression | Hold-out validation accuracy and model coefficients |
| 5. Training accuracy | Accuracy on the full training set |
| 6. Predict on test set | Same preparation steps applied to `test.csv`, number of predicted survivors |
| 7. Test accuracy | See note below |

### Data preparation applied

- **Title** extracted from `Name` (Mr, Mrs, Miss, Master, Rare)
- **Age** missing values filled with the median age of passengers with the same title
- **Fare** and **Embarked** missing values filled with the training median / most common port
- **FamilySize** = SibSp + Parch + 1, and **IsAlone** flag
- **Sex** encoded as 0/1; **Pclass**, **Embarked**, **Title** one-hot encoded
- **PassengerId, Name, Ticket, Cabin** dropped (identifiers, free text, or mostly missing)

All fill values are computed from the training set only and reused on the test set.

### A note on test accuracy

Kaggle's `test.csv` does **not** include the `Survived` column, so true test accuracy cannot be computed locally. The scripts therefore report:

1. **Hold-out validation accuracy** – 20% of `train.csv` is held out, which is the best local estimate of test performance.
2. **Agreement with `gender_submission.csv`** – Kaggle's baseline that predicts all women survive and all men die. This is a reference point, not true accuracy.

For the true test score, upload the generated predictions file to the Kaggle competition.

### Output files

The containers write predictions back into your local `src/data/` folder (it is mounted, and git-ignored):

- `src/data/predictions_python.csv`
- `src/data/predictions_r.csv`

Both are in Kaggle's submission format (`PassengerId,Survived`).

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `[ERROR] /app/data/train.csv not found` | The CSVs are not in `src/data/`, or you ran `docker run` from a different folder. `cd` into the repo root and re-check Step 2. |
| `Cannot connect to the Docker daemon` | Start Docker Desktop and wait until it says it is running. |
| Volume path errors on Windows | Use the PowerShell or Command Prompt command for your shell exactly as shown. |
| R build is slow | Normal on the first build; later builds reuse the cached layers. |

## Package versions

- **Python 3.11:** numpy 2.1.3, pandas 2.2.3, scikit-learn 1.5.2 (pinned in `src/python/requirements.txt`)
- **R 4.4.1:** readr, dplyr – installed from the dated CRAN snapshot that the `rocker/r-ver:4.4.1` image is pinned to, so versions are fixed. The exact versions are printed during `docker build`.
