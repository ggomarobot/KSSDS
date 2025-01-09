import os
import yaml
from .train import train_model
from .evaluate import evaluate_model

def collect_files(folder_path, file_extension):
    """
    Collect all files from a folder and its subdirectories that match the given extension.
    """
    collected_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(file_extension):
                collected_files.append(os.path.join(root, file))
    return collected_files

if __name__ == "__main__":
    # Load configuration from the YAML file
    with open("./config/main_config.yaml", "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    # Get dataset folder paths from the config
    training_dataset_folder = config["data"]["training_dataset_folder"]
    validation_dataset_folder = config["data"]["validation_dataset_folder"]
    test_dataset_folder = config["data"]["test_dataset_folder"]
    
    src_file_extension = config["data"]["src_file_extension"]

    # Collect dataset files using the modular function
    training_dataset_files = collect_files(training_dataset_folder, src_file_extension)
    validation_dataset_files = collect_files(validation_dataset_folder, src_file_extension)
    test_dataset_files = collect_files(test_dataset_folder, src_file_extension)

    # Train the model
    trainer = train_model(training_dataset_files, validation_dataset_files, config)

    # Perform final evaluation using the trained model
    evaluate_model(trainer, test_dataset_files)

