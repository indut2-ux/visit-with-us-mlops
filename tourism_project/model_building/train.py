
import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)


# =========================================================
# 1. FILE PATHS
# =========================================================

TRAIN_PATH = "tourism_project/model_building/train.csv"
TEST_PATH = "tourism_project/model_building/test.csv"

DEPLOYMENT_DIR = "tourism_project/deployment"
MODEL_PATH = os.path.join(DEPLOYMENT_DIR, "best_model.pkl")

os.makedirs(DEPLOYMENT_DIR, exist_ok=True)


# =========================================================
# 2. LOAD TRAIN AND TEST DATA
# =========================================================

print("=" * 60)
print("MODEL BUILDING WITH EXPERIMENT TRACKING")
print("=" * 60)

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print("\nTraining data loaded successfully:", train_df.shape)
print("Testing data loaded successfully :", test_df.shape)


# =========================================================
# 3. SEPARATE FEATURES AND TARGET
# =========================================================

TARGET = "ProdTaken"

X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]

X_test = test_df.drop(columns=[TARGET])
y_test = test_df[TARGET]

print("\nTraining features:", X_train.shape)
print("Testing features :", X_test.shape)


# =========================================================
# 4. IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# =========================================================

numeric_features = X_train.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object", "category"]
).columns.tolist()

print("\nNumerical Features:")
print(numeric_features)

print("\nCategorical Features:")
print(categorical_features)


# =========================================================
# 5. DATA PREPROCESSING
# =========================================================

# Numerical missing values -> median
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

# Categorical missing values -> most frequent
# followed by One-Hot Encoding
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


# =========================================================
# 6. DEFINE RANDOM FOREST MODEL
# =========================================================

random_forest = RandomForestClassifier(
    random_state=42,
    class_weight="balanced"
)


# =========================================================
# 7. CREATE ML PIPELINE
# =========================================================

model_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", random_forest)
    ]
)


# =========================================================
# 8. DEFINE HYPERPARAMETERS
# =========================================================

param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [None, 10, 20],
    "classifier__min_samples_split": [2, 5]
}

print("\nHyperparameter Grid:")
print(param_grid)


# =========================================================
# 9. CONFIGURE MLFLOW
# =========================================================



mlflow.set_tracking_uri("sqlite:///mlflow.db")

mlflow.set_experiment(
    "Tourism_Package_Prediction"
)


# =========================================================
# 10. START MLFLOW EXPERIMENT
# =========================================================

with mlflow.start_run(run_name="Random_Forest_GridSearch"):

    print("\nStarting GridSearchCV...")

    # -----------------------------------------------------
    # Hyperparameter tuning
    # -----------------------------------------------------

    grid_search = GridSearchCV(
        estimator=model_pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    print("\nHyperparameter tuning completed!")


    # -----------------------------------------------------
    # Best model and parameters
    # -----------------------------------------------------

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    print("\nBEST PARAMETERS")
    print("-" * 40)

    for parameter, value in best_params.items():
        print(parameter, ":", value)


    # -----------------------------------------------------
    # Log tuned parameter search space
    # -----------------------------------------------------

    mlflow.log_param(
        "n_estimators_options",
        str(param_grid["classifier__n_estimators"])
    )

    mlflow.log_param(
        "max_depth_options",
        str(param_grid["classifier__max_depth"])
    )

    mlflow.log_param(
        "min_samples_split_options",
        str(param_grid["classifier__min_samples_split"])
    )


    # -----------------------------------------------------
    # Log best tuned parameters
    # -----------------------------------------------------

    for parameter, value in best_params.items():
        mlflow.log_param(parameter, value)


    # -----------------------------------------------------
    # 11. EVALUATE BEST MODEL
    # -----------------------------------------------------

    predictions = best_model.predict(X_test)

    probabilities = best_model.predict_proba(
        X_test
    )[:, 1]

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )


    # -----------------------------------------------------
    # 12. LOG METRICS TO MLFLOW
    # -----------------------------------------------------

    mlflow.log_metric(
        "accuracy",
        accuracy
    )

    mlflow.log_metric(
        "precision",
        precision
    )

    mlflow.log_metric(
        "recall",
        recall
    )

    mlflow.log_metric(
        "f1_score",
        f1
    )

    mlflow.log_metric(
        "roc_auc",
        roc_auc
    )


    # -----------------------------------------------------
    # 13. LOG MODEL TO MLFLOW
    # -----------------------------------------------------

    mlflow.sklearn.log_model(
        sk_model=best_model,
        name="random_forest_model"
    )


    # -----------------------------------------------------
    # 14. SAVE BEST MODEL
    # -----------------------------------------------------

    joblib.dump(
        best_model,
        MODEL_PATH
    )


    # -----------------------------------------------------
    # 15. PRINT PERFORMANCE
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("BEST MODEL PERFORMANCE")
    print("=" * 60)

    print(
        "Accuracy :",
        round(accuracy, 4)
    )

    print(
        "Precision:",
        round(precision, 4)
    )

    print(
        "Recall   :",
        round(recall, 4)
    )

    print(
        "F1 Score :",
        round(f1, 4)
    )

    print(
        "ROC-AUC  :",
        round(roc_auc, 4)
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions
        )
    )

    print("\nBest model saved successfully:")
    print(MODEL_PATH)


print("\n" + "=" * 60)
print(
    "MODEL TRAINING AND EXPERIMENT TRACKING COMPLETED!"
)
print("=" * 60)
