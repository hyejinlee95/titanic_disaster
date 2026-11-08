"""
Titanic survival prediction with logistic regression (Python).

Reads train.csv / test.csv from DATA_DIR (default: /app/data, mounted from src/data),
prints every data-preparation step, fits a logistic regression, reports accuracy,
and writes predictions to DATA_DIR/predictions_python.csv.
"""

import os
import sys

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
RANDOM_STATE = 42


def banner(text):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def load_csv(name):
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        print(f"[ERROR] {path} not found.")
        print("        Download the Kaggle Titanic data and place it in src/data/ (see README).")
        sys.exit(1)
    df = pd.read_csv(path)
    print(f"Loaded {name}: {df.shape[0]} rows x {df.shape[1]} columns")
    return df


def extract_title(name_series):
    titles = name_series.str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()
    titles = titles.replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    common = {"Mr", "Mrs", "Miss", "Master"}
    return titles.where(titles.isin(common), "Rare")


def prepare(df, fill_values, label):
    """Apply the same cleaning / feature engineering to any dataframe, printing each change."""
    df = df.copy()
    print(f"\n--- Preparing {label} set ---")

    df["Title"] = extract_title(df["Name"])
    print(f"[ADD] Title extracted from Name -> {df['Title'].value_counts().to_dict()}")

    n = df["Age"].isna().sum()
    df["Age"] = df.apply(
        lambda r: fill_values["age_by_title"].get(r["Title"], fill_values["age"])
        if pd.isna(r["Age"]) else r["Age"],
        axis=1,
    )
    print(f"[FILL] Age: {n} missing values filled with the training median age per Title")

    n = df["Fare"].isna().sum()
    df["Fare"] = df["Fare"].fillna(fill_values["fare"])
    print(f"[FILL] Fare: {n} missing values filled with training median ({fill_values['fare']:.2f})")

    n = df["Embarked"].isna().sum()
    df["Embarked"] = df["Embarked"].fillna(fill_values["embarked"])
    print(f"[FILL] Embarked: {n} missing values filled with training mode ('{fill_values['embarked']}')")

    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    print(f"[ADD] FamilySize = SibSp + Parch + 1 (mean {df['FamilySize'].mean():.2f})")

    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    print(f"[ADD] IsAlone = 1 if FamilySize == 1 ({df['IsAlone'].sum()} passengers travelling alone)")

    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    print("[ENCODE] Sex: male -> 0, female -> 1")

    dropped = ["PassengerId", "Name", "Ticket", "Cabin"]
    print(f"[DROP] {dropped} (identifiers / free text / >75% missing Cabin)")

    features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare",
                "FamilySize", "IsAlone", "Embarked", "Title"]
    X = pd.get_dummies(df[features], columns=["Pclass", "Embarked", "Title"], drop_first=True, dtype=int)
    print(f"[ENCODE] One-hot encoded Pclass, Embarked, Title -> {X.shape[1]} feature columns")
    return X


def main():
    banner("STEP 1: Load training data")
    train = load_csv("train.csv")
    print("\nFirst 5 rows:")
    print(train.head().to_string())

    banner("STEP 2: Explore training data")
    print("Missing values per column:")
    print(train.isna().sum()[train.isna().sum() > 0].to_string())
    print(f"\nOverall survival rate: {train['Survived'].mean():.3f}")
    print("\nSurvival rate by Sex:")
    print(train.groupby("Sex")["Survived"].mean().round(3).to_string())
    print("\nSurvival rate by Pclass:")
    print(train.groupby("Pclass")["Survived"].mean().round(3).to_string())

    banner("STEP 3: Clean and engineer features (fill values learned from training data only)")
    train_titles = extract_title(train["Name"])
    fill_values = {
        "age": train["Age"].median(),
        "age_by_title": train.groupby(train_titles)["Age"].median().to_dict(),
        "fare": train["Fare"].median(),
        "embarked": train["Embarked"].mode()[0],
    }
    X_train = prepare(train, fill_values, "training")
    y_train = train["Survived"]
    print(f"\nFinal training features: {list(X_train.columns)}")

    banner("STEP 4: Fit logistic regression")
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))

    # Honest estimate: hold out 20% of train.csv
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=RANDOM_STATE, stratify=y_train)
    model.fit(X_tr, y_tr)
    val_acc = accuracy_score(y_val, model.predict(X_val))
    print(f"Hold-out validation accuracy (20% of train.csv): {val_acc:.4f}")

    # Final model on all training data
    model.fit(X_train, y_train)
    coefs = pd.Series(model[-1].coef_[0], index=X_train.columns).sort_values()
    print("\nStandardized coefficients (negative = lower survival odds):")
    print(coefs.round(3).to_string())

    banner("STEP 5: Accuracy on the training set")
    train_acc = accuracy_score(y_train, model.predict(X_train))
    print(f"Training accuracy: {train_acc:.4f}")

    banner("STEP 6: Load test data and predict")
    test = load_csv("test.csv")
    X_test = prepare(test, fill_values, "test")
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    print("[ALIGN] Test columns aligned to training columns")
    test_pred = model.predict(X_test)
    print(f"\nPredicted survivors in test set: {test_pred.sum()} of {len(test_pred)} "
          f"({test_pred.mean():.3f})")

    out_path = os.path.join(DATA_DIR, "predictions_python.csv")
    pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": test_pred}).to_csv(out_path, index=False)
    print(f"Predictions saved to {out_path}")

    banner("STEP 7: Accuracy on the test set")
    if "Survived" in test.columns:
        print(f"Test accuracy: {accuracy_score(test['Survived'], test_pred):.4f}")
    else:
        print("Kaggle's test.csv has no 'Survived' column, so true test accuracy cannot be computed locally.")
        gs_path = os.path.join(DATA_DIR, "gender_submission.csv")
        if os.path.exists(gs_path):
            gs = pd.read_csv(gs_path)
            merged = test[["PassengerId"]].assign(Pred=test_pred).merge(gs, on="PassengerId")
            agree = (merged["Pred"] == merged["Survived"]).mean()
            print(f"Agreement with Kaggle's gender_submission.csv baseline: {agree:.4f}")
            print("(This is agreement with a baseline, NOT true accuracy.)")
        print(f"Best local estimate of test accuracy = hold-out validation accuracy: {val_acc:.4f}")
        print("For the true score, upload predictions_python.csv to Kaggle.")

    banner("DONE")


if __name__ == "__main__":
    main()
