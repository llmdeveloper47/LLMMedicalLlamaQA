import yaml
from evaluation.model_evaluation import Evaluator

def load_config(config_path: str) -> dict:
    """
    Load configuration from a YAML file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: Parsed configuration as a dictionary.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def run_base_evaluation(config):

    if config['evaluation']['base_model_evaluation'] == True:

        model_path = config['evaluation']['model_artifacts_path']
        dataset_name = config['evaluation']['dataset_']


        Evaluator( )









# Example usage
if __name__ == "__main__":
    config_path = "./config/config.yaml"  # Path to your config.yaml file
    config = load_config(config_path)

    # Accessing config values
    model_path = config['inference']['model_path']
    tokenizer_path = config['inference']['tokenizer_path']
    logging_level = config['logging']['level']

    print("Model Path:", model_path)
    print("Tokenizer Path:", tokenizer_path)
    print("Logging Level:", logging_level)