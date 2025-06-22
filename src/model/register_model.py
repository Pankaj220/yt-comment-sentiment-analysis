# register model
import json
import mlflow
import logging
import os

# Set up MLflow tracking URI
mlflow.set_tracking_uri("http://ec2-54-157-11-41.compute-1.amazonaws.com:5000/")

# logging configuration
logger = logging.getLogger('model_registration')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_registration_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logger.debug('Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logger.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the model info: %s', e)
        raise

def register_model(model_name: str, model_info: dict):
    """Register the model to the MLflow Model Registry."""
    try:
        # Use the standard MLflow runs URI format (this is the preferred way)
        model_uri = f"runs:/{model_info['run_id']}/lgbm_model"
        
        logger.debug(f'Registering model with URI: {model_uri}')
        
        # Register the model
        model_version = mlflow.register_model(model_uri, model_name)
        
        logger.debug(f'Model {model_name} version {model_version.version} registered successfully.')
        
        # Transition the model to "Staging" stage
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )
        
        logger.debug(f'Model {model_name} version {model_version.version} transitioned to Staging.')
        print(f'Model {model_name} version {model_version.version} registered and transitioned to Staging successfully!')
        
    except Exception as e:
        logger.error('Error during model registration: %s', e)
        raise

def main():
    try:
        # Get the root directory to find experiment_info.json
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
        model_info_path = os.path.join(root_dir, 'experiment_info.json')
        
        # Check if file exists
        if not os.path.exists(model_info_path):
            logger.error(f'Model info file not found at: {model_info_path}')
            print(f'Error: Model info file not found at: {model_info_path}')
            return
        
        model_info = load_model_info(model_info_path)
        
        # model_name = "yt_chrome_plugin_model"
        model_name = "my_model"
        register_model(model_name, model_info)
        
    except Exception as e:
        logger.error('Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()