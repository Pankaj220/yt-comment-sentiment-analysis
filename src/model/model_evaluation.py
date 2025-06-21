import numpy as np
import pandas as pd
import pickle
import logging
import yaml
import mlflow
import mlflow.sklearn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import matplotlib.pyplot as plt
import seaborn as sns
import json

# logging configuration
logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')
console_handler = logging.StreamHandler(); console_handler.setLevel('DEBUG')
file_handler = logging.FileHandler('model_evaluation_errors.log'); file_handler.setLevel('ERROR')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter); file_handler.setFormatter(formatter)
logger.addHandler(console_handler); logger.addHandler(file_handler)

def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path); df.fillna('', inplace=True)
        logger.debug('Data loaded and NaNs filled from %s', file_path)
        return df
    except Exception as e:
        logger.error('Error loading data from %s: %s', file_path, e)
        raise

def load_model(model_path: str):
    try:
        with open(model_path, 'rb') as f: model = pickle.load(f)
        logger.debug('Model loaded from %s', model_path)
        return model
    except Exception as e:
        logger.error('Error loading model from %s: %s', model_path, e)
        raise

def load_vectorizer(vectorizer_path: str) -> TfidfVectorizer:
    try:
        with open(vectorizer_path, 'rb') as f: vectorizer = pickle.load(f)
        logger.debug('TF-IDF vectorizer loaded from %s', vectorizer_path)
        return vectorizer
    except Exception as e:
        logger.error('Error loading vectorizer from %s: %s', vectorizer_path, e)
        raise

def load_params(params_path: str) -> dict:
    try:
        with open(params_path, 'r') as f: params = yaml.safe_load(f)
        logger.debug('Parameters loaded from %s', params_path)
        return params
    except Exception as e:
        logger.error('Error loading parameters from %s: %s', params_path, e)
        raise

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray):
    try:
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        logger.debug('Model evaluation completed')
        return report, cm, y_pred  # Added y_pred to return
    except Exception as e:
        logger.error('Error during model evaluation: %s', e)
        raise

def log_confusion_matrix(cm, dataset_name):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {dataset_name}')
    plt.xlabel('Predicted'); plt.ylabel('Actual')
    cm_file = f'confusion_matrix_{dataset_name.replace(" ", "_").lower()}.png'
    plt.savefig(cm_file); plt.close()
    if os.path.isfile(cm_file):
        mlflow.log_artifact(cm_file, artifact_path="confusion_matrix")
        logger.debug('Logged confusion matrix artifact: %s', cm_file)
    else:
        logger.error('Confusion matrix file not found: %s', cm_file)

def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    try:
        info = {'run_id': run_id, 'model_path': model_path}
        with open(file_path, 'w') as f: json.dump(info, f, indent=4)
        logger.debug('Model info saved to %s', file_path)
    except Exception as e:
        logger.error('Error saving model info: %s', e)
        raise

def main():
    os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/Pankaj220/yt-comment-sentiment-analysis.mlflow"
    os.environ["MLFLOW_TRACKING_USERNAME"] = "pankaj220"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "fbae2cdee75ebf744f5768abec50130e1462efa7"

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment('dvc-pipeline-runs')
    run = mlflow.start_run()

    try:
        # Load from current directory (matching your DVC structure)
        params = load_params('params.yaml')
        for k, v in params.items(): mlflow.log_param(k, v)

        model = load_model('lgbm_model.pkl')
        vect  = load_vectorizer('tfidf_vectorizer.pkl')
        if hasattr(model, 'get_params'):
            for p, val in model.get_params().items():
                mlflow.log_param(p, val)

        # Log artifacts FIRST before any potential failures
        logger.info("Starting to log model and vectorizer...")
        try:
            mlflow.log_artifact('tfidf_vectorizer.pkl', artifact_path="vectorizer")
            logger.info("✅ Vectorizer logged successfully")
        except Exception as e:
            logger.error(f"Failed to log vectorizer: {e}")
            
        # Try to log model - but continue even if it fails
        try:
            mlflow.sklearn.log_model(sk_model=model, artifact_path="lgbm_model")
            logger.info("✅ Model logged successfully")
        except Exception as e:
            logger.error(f"Failed to log model: {e}")
            logger.info("Continuing without model logging...")

        # Load both train and test data as per your DVC deps
        logger.info("Loading data...")
        train_data = load_data('data/interim/train_processed.csv')
        data = load_data('data/interim/test_processed.csv')
        X_test = vect.transform(data['clean_comment'].values)
        y_test = data['category'].values
        logger.info(f"Data loaded: train={len(train_data)}, test={len(y_test)}")

        # Evaluate model
        logger.info("Evaluating model...")
        report, cm, y_pred = evaluate_model(model, X_test, y_test)
        logger.info("✅ Model evaluation completed")
        
        # Log confusion matrix IMMEDIATELY
        logger.info("Creating confusion matrix...")
        try:
            log_confusion_matrix(cm, "Test_Data")
            logger.info("✅ Confusion matrix logged")
        except Exception as e:
            logger.error(f"Failed to log confusion matrix: {e}")
        
        # Log all metrics from classification report
        logger.info("Logging metrics...")
        try:
            for lbl, mets in report.items():
                if isinstance(mets, dict):
                    for metric_name, metric_value in mets.items():
                        mlflow.log_metric(f"test_{lbl}_{metric_name}", metric_value)
            
            # Log overall accuracy
            if 'accuracy' in report:
                mlflow.log_metric("test_accuracy", report['accuracy'])
            logger.info("✅ Metrics logged successfully")
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")
        
        # Save and log evaluation metrics as JSON
        logger.info("Saving evaluation metrics...")
        try:
            eval_metrics = {
                'classification_report': report,
                'confusion_matrix': cm.tolist(),
                'test_accuracy': report.get('accuracy', 0),
                'macro_avg_f1': report.get('macro avg', {}).get('f1-score', 0),
                'weighted_avg_f1': report.get('weighted avg', {}).get('f1-score', 0)
            }
            
            # Save metrics to file and log as artifact
            metrics_file = 'evaluation_metrics.json'
            with open(metrics_file, 'w') as f:
                json.dump(eval_metrics, f, indent=4)
            mlflow.log_artifact(metrics_file, artifact_path="metrics")
            logger.info("✅ Evaluation metrics JSON logged")
        except Exception as e:
            logger.error(f"Failed to log evaluation metrics: {e}")
        
        # Log predictions as artifact
        logger.info("Saving predictions...")
        try:
            predictions_file = 'predictions.json'
            pred_data = {
                'true_labels': y_test.tolist(),
                'predicted_labels': y_pred.tolist(),
                'sample_texts': data['clean_comment'].head(100).tolist()  # Log first 100 samples
            }
            with open(predictions_file, 'w') as f:
                json.dump(pred_data, f, indent=4)
            mlflow.log_artifact(predictions_file, artifact_path="predictions")
            logger.info("✅ Predictions logged")
        except Exception as e:
            logger.error(f"Failed to log predictions: {e}")

        # Set tags
        logger.info("Setting tags...")
        try:
            mlflow.set_tag("model_type", "LightGBM")
            mlflow.set_tag("task", "Sentiment Analysis")
            mlflow.set_tag("dataset", "YouTube Comments")
            mlflow.set_tag("train_samples", len(train_data))
            mlflow.set_tag("test_samples", len(y_test))
            mlflow.set_tag("features", X_test.shape[1])
            logger.info("✅ Tags set successfully")
        except Exception as e:
            logger.error(f"Failed to set tags: {e}")

        # Final debug - list all artifacts
        try:
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            arts = client.list_artifacts(run.info.run_id)
            logger.info(f"Final artifacts logged: {[f.path for f in arts]}")
        except Exception as e:
            logger.error(f"Failed to list artifacts: {e}")

        print(f"✅ All artifacts logged successfully!")
        if eval_metrics:
            print(f"📊 Test Accuracy: {eval_metrics['test_accuracy']:.4f}")
            print(f"🎯 Macro F1-Score: {eval_metrics['macro_avg_f1']:.4f}")

    except Exception as e:
        logger.error("Failed to complete model evaluation: %s", e)
        print(f"Error: {e}")

    finally:
        try:
            uri = mlflow.get_artifact_uri()
            mp  = f"{uri}/lgbm_model"
            rid = run.info.run_id if run else "unknown"
            save_model_info(rid, mp, 'experiment_info.json')
            if os.path.isfile('experiment_info.json'):
                mlflow.log_artifact('experiment_info.json', artifact_path="run_info")
                logger.debug('Logged experiment_info.json artifact')
            else:
                logger.error('experiment_info.json not found to log')
        except Exception as e:
            logger.error("Could not save fallback experiment_info.json: %s", e)
        mlflow.end_run()

if __name__ == '__main__':
    main()