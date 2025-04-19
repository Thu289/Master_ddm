import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

parser = argparse.ArgumentParser()

parser.add_argument("--samples", type=int, default=1000)
parser.add_argument("--features", type=int, default=20)
parser.add_argument("--classes", type=int, default=2)
parser.add_argument("--informative", type=int, default=10)
parser.add_argument("--state", type=int, default=42)
parser.add_argument("--max_depth", type=int, default=3)
parser.add_argument("--estimators", type=int, default=100)
parser.add_argument("--learning_rate", type=float, default=0.1)
parser.add_argument("--regularization_strength", type=float, default=1.0)
args = parser.parse_args()
experiment_name = "classification-synthetic-data"

# Define experiment name
if mlflow.get_experiment_by_name(experiment_name) is None:
    mlflow.create_experiment(experiment_name)

# Set or create the experiment
mlflow.set_experiment(experiment_name)

def generate_data(samples, features, classes, informative, random_state):
    x,y = make_classification(
        n_samples=samples,
        n_features=features,
        n_classes=classes,
        n_informative=informative,
        random_state=random_state,
    )

    feature_names = [f'f{i}' for i in range(features)]
    x_df = pd.DataFrame(x, columns=feature_names)
    y_series = pd.Series(y, name="target")

    return x_df, y_series


def train_random_forest(X_train, y_train, X_test, y_test, max_depth, n_estimators):
    with mlflow.start_run(run_name="random_forest"):
        # Log parameters
        mlflow.log_params({
            "n_samples": args.samples,
            "n_features": args.features,
            "n_classes": args.classes,
            "n_informative": args.informative,
            "max_depth": max_depth,
            "n_estimators": n_estimators,
            "model_type": "RandomForest"
        })

        # Build model
        model = RandomForestClassifier(
            max_depth=max_depth,
            n_estimators=n_estimators,
            random_state=args.state
        )

        # Train model
        model.fit(X_train, y_train)

        # Evaluate model
        metrics, cm, y_pred = evaluate_model(model, X_test, y_test)

        # Plot confusion matrix
        classes = np.unique(y_test)
        cm_plot_path = plot_confusion_matrix(cm, classes, "RandomForest")
        mlflow.log_artifact(cm_plot_path)

        # Log metrics
        for key, value in metrics.items():
            mlflow.log_metric(key, value)

        # Save the feature importances
        feature_importances = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)

        fi_path = "feature_importances_rf.csv"
        feature_importances.to_csv(fi_path, index=False)
        mlflow.log_artifact(fi_path)

        # Infer model signature
        signature = infer_signature(X_train, model.predict(X_train))

        # Log model
        mlflow.sklearn.log_model(model, "random_forest_model", signature=signature)

        return metrics

def train_logistic_regression(X_train, y_train, X_test, y_test, regularization_strength):
    """Train and evaluate Logistic Regression classifier"""
    with mlflow.start_run(run_name="logistic_regression"):
        # Log parameters
        mlflow.log_params({
            "n_samples": args.samples,
            "n_features": args.features,
            "n_classes": args.classes,
            "n_informative": args.informative,
            "regularization_strength": regularization_strength,
            "model_type": "LogisticRegression"
        })
        
        # Build model
        model = LogisticRegression(
            C=regularization_strength,
            max_iter=1000,
            random_state=args.state
        )
        
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate model
        metrics, cm, y_pred = evaluate_model(model, X_test, y_test)
        
        # Plot confusion matrix
        classes = np.unique(y_test)
        cm_plot_path = plot_confusion_matrix(cm, classes, "LogisticRegression")
        mlflow.log_artifact(cm_plot_path)
        
        # Log metrics
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
        
        # Save the coefficients
        if hasattr(model, 'coef_'):
            coef = pd.DataFrame({
                'Feature': X_train.columns,
                'Coefficient': model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
            }).sort_values('Coefficient', ascending=False)
            
            coef_path = "coefficients_lr.csv"
            coef.to_csv(coef_path, index=False)
            mlflow.log_artifact(coef_path)
        
        # Infer model signature
        signature = infer_signature(X_train, model.predict(X_train))
        
        # Log model
        mlflow.sklearn.log_model(model, "logistic_regression_model", signature=signature)
        
        return metrics

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='weighted'),
        "recall": recall_score(y_test, y_pred, average='weighted'),
        "f1": f1_score(y_test, y_pred, average='weighted'),
    }

    return metrics, confusion_matrix(y_test, y_pred), y_pred


def plot_confusion_matrix(cm, classes, model_name):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    plot_path = f"confusion_matrix_{model_name}.png"
    plt.savefig(plot_path)
    return plot_path

def hyperparameter_tuning_rf(x_train, y_train, x_test, y_test):
    with mlflow.start_run(run_name="rf_hyperparameter_tuning"):
        mlflow.log_params({
            "n_samples": args.samples,
            "n_features": args.features,
            "n_classes": args.classes,
            "n_informative": args.informative,
            "model_type": "RandomForest_GridSearch"
        })

        param_grid = {
            'max_depth': [3, 5, 7, 10],
            'n_estimators': [50, 100, 200]
        }

        rf = RandomForestClassifier(random_state=args.state)

        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=3,
            scoring='f1_weighted',
            verbose=1
        )

        grid_search.fit(x_train, y_train)
        best_params = grid_search.best_params_
        mlflow.log_params(best_params)
        best_model = RandomForestClassifier(
            max_depth=best_params['max_depth'],
            n_estimators=best_params['n_estimators'],
            random_state=args.state
        )
        best_model.fit(x_train, y_train)

        metrics, cm, y_pred = evaluate_model(best_model, x_test, y_test)

        classes = np.unique(y_test)
        cm_plot_path = plot_confusion_matrix(cm, classes, "RandomForest_Tuned")
        mlflow.log_artifact(cm_plot_path)

        for key, value in metrics.items():
            mlflow.log_metric(key, value)

        cv_results = pd.DataFrame(grid_search.cv_results_)
        cv_results_path = "rf_grid_search_results.csv"
        cv_results.to_csv(cv_results_path, index=False)
        mlflow.log_artifact(cv_results_path)

        signature = infer_signature(x_train, best_model.predict(x_train))

        mlflow.sklearn.log_model(best_model, "rf_tuned_model", signature=signature)

        return metrics, best_params

def hyperparameter_tuning_lr(X_train, y_train, X_test, y_test):
    """Hyperparameter tuning for Logistic Regression"""
    with mlflow.start_run(run_name="lr_hyperparameter_tuning"):
        # Log base parameters
        mlflow.log_params({
            "n_samples": args.samples,
            "n_features": args.features,
            "n_classes": args.classes,
            "n_informative": args.informative,
            "model_type": "LogisticRegression_GridSearch"
        })
        
        # Define parameter grid
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'solver': ['liblinear', 'lbfgs'],
            'penalty': ['l1', 'l2']
        }
        
        # Create base model
        lr = LogisticRegression(random_state=args.state, max_iter=1000)
        
        # Create grid search
        grid_search = GridSearchCV(
            estimator=lr,
            param_grid=param_grid,
            cv=3,
            scoring='f1_weighted',
            verbose=1
        )
        
        # Fit grid search
        grid_search.fit(X_train, y_train)
        
        # Get best parameters
        best_params = grid_search.best_params_
        mlflow.log_params(best_params)
        
        # Train model with best parameters
        best_model = LogisticRegression(
            C=best_params['C'],
            solver=best_params['solver'],
            penalty=best_params['penalty'],
            max_iter=1000,
            random_state=args.state
        )
        best_model.fit(X_train, y_train)
        
        # Evaluate model
        metrics, cm, y_pred = evaluate_model(best_model, X_test, y_test)
        
        # Plot confusion matrix
        classes = np.unique(y_test)
        cm_plot_path = plot_confusion_matrix(cm, classes, "LogisticRegression_Tuned")
        mlflow.log_artifact(cm_plot_path)
        
        # Log metrics
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
        
        # Save grid search results
        cv_results = pd.DataFrame(grid_search.cv_results_)
        cv_results_path = "lr_grid_search_results.csv"
        cv_results.to_csv(cv_results_path, index=False)
        mlflow.log_artifact(cv_results_path)
        
        # Infer model signature
        signature = infer_signature(X_train, best_model.predict(X_train))
        
        # Log model
        mlflow.sklearn.log_model(best_model, "lr_tuned_model", signature=signature)
        
        return metrics, best_params

def find_best_run(experiment_name):
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if not experiment:
        print(f"No experiment named '{experiment_name}' found.")
        return None

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="",
        order_by=["metrics.f1 DESC"]
    )

    if not runs:
        print("No runs found in the experiment.")
        return None

    best_run = runs[0]
    print(f"Best run ID: {best_run.info.run_id}")
    print(f"Best run metrics: {best_run.data.metrics}")

    model_uri = f"runs:/{best_run.info.run_id}"
    model_name = "best_classification_model"

    model_type = best_run.data.params.get("model_type", "")

    if "RandomForest" in model_type:
        model_path = f"{model_uri}/rf_tuned_model" if "GridSearch" in model_type else f"{model_uri}/random_forest_model"
    else:
        model_path = f"{model_uri}/xgb_tuned_model" if "GridSearch" in model_type else f"{model_uri}/logistic_regression_model"

    registered_model = mlflow.register_model(model_path, model_name)
    print(f"Registered model: {registered_model.name}, version: {registered_model.version}")

    with open("best_run_id.txt", "w") as f:
        f.write(best_run.info.run_id)

    return best_run.info.run_id


if __name__ == '__main__':
    n_samples = 1000
    n_features = 20
    n_classes = 2
    n_informative = 10
    random_state = 42

    print(f"Generating {n_samples} samples with {n_features} features ({n_informative} informative)")
    x, y = generate_data(n_samples, n_features, n_classes, n_informative, random_state)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=random_state)
    print(f"Training data shape: {x_train.shape}, Test data shape: {x_test.shape}")

    data_path = "synthetic_data.csv"
    full_data = pd.concat([x,y], axis=1)
    full_data.to_csv(data_path, index=False)

    print(f"Training baseline models ....")
    rf_metrics = train_random_forest(
        x_train, y_train, x_test, y_test, max_depth=args.max_depth, n_estimators=args.estimators
    )
    lr_metrics = train_logistic_regression(
        x_train, y_train, x_test, y_test, 
        regularization_strength=args.regularization_strength
    )

    print("Performing hyperparameter tuning...")
    rf_tuned_metrics, rf_best_params = hyperparameter_tuning_rf(x_train, y_train, x_test, y_test)
    lr_tuned_metrics, lr_best_params = hyperparameter_tuning_lr(x_train, y_train, x_test, y_test)

    # Compare results
    print("\nModel Performance Comparison:")
    print(f"RandomForest Baseline: F1 = {rf_metrics['f1']:.4f}")
    print(f"LogisticRegression Baseline: F1 = {lr_metrics['f1']:.4f}")
    print(f"RandomForest Tuned: F1 = {rf_tuned_metrics['f1']:.4f}, Best Params: {rf_best_params}")
    print(f"LogisticRegression Tuned: F1 = {lr_tuned_metrics['f1']:.4f}, Best Params: {lr_best_params}")

    # Find and register the best model
    print("\nFinding and registering the best model...")
    find_best_run(experiment_name)

    print("\nExperiment completed!")
