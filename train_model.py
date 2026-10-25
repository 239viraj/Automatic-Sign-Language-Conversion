
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# SETTINGS
# ==========================================

DATA_FILE = "sign_data_2hand.csv"
MODEL_FILE = "sign_model.pkl"

EXPECTED_FEATURES = 126

EXPECTED_SIGNS = [
    "HELLO",
    "NAMASTE",
    "GOOD_MORNING",
    "GOOD_NIGHT",
    "GOODBYE",
    "YES",
    "NO",
    "PLEASE",
    "SORRY",
    "THANK_YOU",
    "WELCOME",
    "HELP",
    "STOP",
    "COME",
    "GO",
    "EAT",
    "DRINK",
    "WATER",
    "FOOD",
    "SLEEP",
    "HOME",
    "SCHOOL",
    "FRIEND",
    "FAMILY",
    "HAPPY",
    "SAD",
    "LOVE",
    "UNDERSTAND",
    "GOOD",
    "BAD"
]

# ==========================================
# LOAD DATASET
# ==========================================

if not os.path.exists(DATA_FILE):
    raise SystemExit(
        f"ERROR: {DATA_FILE} not found. Run collect_data.py first."
    )

data = pd.read_csv(DATA_FILE)

print("Dataset loaded successfully!")
print("Total samples:", len(data))

# ==========================================
# VALIDATE DATASET
# ==========================================

if "label" not in data.columns:
    raise SystemExit("ERROR: Dataset must contain a label column.")

X = data.drop(columns=["label"])
y = data["label"]

if X.shape[1] != EXPECTED_FEATURES:
    raise SystemExit(
        f"ERROR: Expected {EXPECTED_FEATURES} features, "
        f"but found {X.shape[1]}."
    )

if data.empty:
    raise SystemExit("ERROR: Dataset is empty.")

if X.isnull().any().any() or y.isnull().any():
    raise SystemExit("ERROR: Dataset contains missing values.")

# Ensure all feature columns are numeric.
try:
    X = X.apply(pd.to_numeric, errors="raise")
except (ValueError, TypeError):
    raise SystemExit("ERROR: Landmark features must be numeric.")

if not X.map(lambda value: pd.notna(value) and abs(value) < float("inf")).all().all():
    raise SystemExit("ERROR: Dataset contains invalid numeric values.")

print("\nSamples per sign:")
print(y.value_counts().sort_index())

# ==========================================
# CHECK ALL 30 SIGNS
# ==========================================

found_signs = set(y.unique())
expected_signs = set(EXPECTED_SIGNS)

missing_signs = sorted(expected_signs - found_signs)
unexpected_signs = sorted(found_signs - expected_signs)

if missing_signs:
    print("\nERROR: These signs have no samples:")
    for sign in missing_signs:
        print("-", sign)

    raise SystemExit(
        "\nCollect samples for all 30 signs before training."
    )

if unexpected_signs:
    print("\nERROR: Unexpected sign labels found:")
    for sign in unexpected_signs:
        print("-", sign)

    raise SystemExit(
        "\nCorrect the dataset labels before training."
    )

class_counts = y.value_counts()

if class_counts.min() < 2:
    raise SystemExit(
        "ERROR: Each sign needs at least two samples."
    )

if class_counts.min() < 30:
    print(
        "\nWARNING: Some signs have fewer than 30 samples. "
        "Their predictions may be unreliable."
    )

# ==========================================
# TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# ==========================================
# CREATE MODEL
# ==========================================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

# ==========================================
# TRAIN MODEL
# ==========================================

print("\nTraining two-hand recognition model...")

model.fit(X_train, y_train)

print("Training completed!")

# ==========================================
# EVALUATE MODEL
# ==========================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n========================================")
print(f"MODEL ACCURACY: {accuracy * 100:.2f}%")
print("========================================")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=EXPECTED_SIGNS,
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred,
        labels=EXPECTED_SIGNS
    )
)

# ==========================================
# SAVE MODEL
# ==========================================

joblib.dump(model, MODEL_FILE)

print("\nModel saved as:", MODEL_FILE)
print("Feature count:", model.n_features_in_)
print("Number of signs:", len(model.classes_))
print("Signs:", list(model.classes_))

