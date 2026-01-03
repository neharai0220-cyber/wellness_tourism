# for data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score
# for model serialization
import joblib
import os
from huggingface_hub import HfApi, create_repo, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("mlops-training-experiment")

api = HfApi()

dataset_repo_id = "NehaRai22/Wellness_Tourism_Prediction"

# Download files locally before reading with pandas to avoid hf:// path issues
# Force download to ensure latest versions are fetched
Xtrain_path_local = hf_hub_download(repo_id=dataset_repo_id, filename="Xtrain.csv", repo_type="dataset", force_download=True)
Xtest_path_local = hf_hub_download(repo_id=dataset_repo_id, filename="Xtest.csv", repo_type="dataset", force_download=True)
ytrain_path_local = hf_hub_download(repo_id=dataset_repo_id, filename="ytrain.csv", repo_type="dataset", force_download=True)
ytest_path_local = hf_hub_download(repo_id=dataset_repo_id, filename="ytest.csv", repo_type="dataset", force_download=True)

Xtrain = pd.read_csv(Xtrain_path_local)
Xtest = pd.read_csv(Xtest_path_local)
ytrain = pd.read_csv(ytrain_path_local).squeeze()
ytest = pd.read_csv(ytest_path_local).squeeze()

print(f"Shape of Xtrain loaded in train.py: {Xtrain.shape}")
print(f"Shape of ytrain loaded in train.py: {ytrain.shape}")

# Define numeric and categorical features (these should be consistent with prep.py)
numeric_features = [
        "Age",
        "NumberOfPersonVisiting",
        "NumberOfTrips",
        "NumberOfChildrenVisiting",
        "NumberOfFollowups",
        "DurationOfPitch",
        "PitchSatisfactionScore",
        "MonthlyIncome",
  ]

categorical_features = [
        "TypeofContact",
        "CityTier",
        "Occupation",
        "Gender",
        "ProductPitched",
        "PreferredPropertyStar",
        "MaritalStatus",
        "Designation",
    ]

# Define the preprocessing steps
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# Define base XGBoost model
xgb_model = xgb.XGBClassifier(random_state=42)

# Define hyperparameter grid
param_grid = {
    'xgbclassifier__n_estimators': [50, 75, 100],
    'xgbclassifier__max_depth': [2, 3, 4],
    'xgbclassifier__colsample_bytree': [0.4, 0.5, 0.6],
    'xgbclassifier__colsample_bylevel': [0.4, 0.5, 0.6],
    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],
    'xgbclassifier__reg_lambda': [0.4, 0.5, 0.6],
}

# Model pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

# Start MLflow run
with mlflow.start_run():
    # Hyperparameter tuning
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(Xtrain, ytrain) # This is where the error occurs

    # Log best parameters separately in main run
    mlflow.log_params(grid_search.best_params_)

    # Log the full cross-validation results as an artifact
    cv_results_df = pd.DataFrame(grid_search.cv_results_)
    cv_results_df.to_csv("cv_results.csv", index=False)
    mlflow.log_artifact("cv_results.csv", artifact_path="grid_search_results")
    os.remove("cv_results.csv") # Clean up local file

    # Store and evaluate the best model
    best_model = grid_search.best_estimator_

    classification_threshold = 0.45

    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    # Log the metrics for the best model
    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision": train_report['1']['precision'],
        "train_recall": train_report['1']['recall'],
        "train_f1-score": train_report['1']['f1-score'],
        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report['1']['precision'],
        "test_recall": test_report['1']['recall'],
        "test_f1-score": test_report['1']['f1-score']
    })

    # Save the model locally
    model_path = "best_tourism_ProdTaken_v1.joblib"
    joblib.dump(best_model, model_path)

    # Log the model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

    # Upload to Hugging Face
    repo_id = "NehaRai22/tourism_package_prediction_model"
    repo_type = "model"

    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Space '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Space '{repo_id}' not found. Creating new space...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Space '{repo_id}' created.")

    api.upload_file(
        path_or_fileobj="best_tourism_ProdTaken_v1.joblib",
        path_in_repo="best_tourism_ProdTaken_v1.joblib",
        repo_id=repo_id,
        repo_type=repo_type,
    )
