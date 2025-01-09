import os
import yaml
from transformers import AutoTokenizer, TrainingArguments
from .dataloader import CustomDataLoader, custom_collate_fn
from .dataset import RawCustomDataset
from .trainer import CustomTrainer
from .T5_encoder import T5ForTokenClassification
from .util import compute_metrics, CombinedTensorboardCallback

def initialize_trainer(model, tokenizer, config, eval_dataset=None):
    """
    Initialize a Hugging Face Trainer for evaluation.
    """
    training_args = TrainingArguments(
        output_dir=config["training"]["output_dir"],
        per_device_eval_batch_size=config["training"]["per_device_eval_batch_size"],
        logging_dir=config["training"]["logging_dir"],
        remove_unused_columns=config["training"]["remove_unused_columns"],
        report_to=config["training"]["report_to"],
    )

    return CustomTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        eval_dataset=eval_dataset,
        data_collator=custom_collate_fn,
        compute_metrics=compute_metrics,
        callbacks=[CombinedTensorboardCallback],
        custom_config=config["dataloader"]
    )

def load_eval_dataset(file_paths, config=None):
    """
    Create an evaluation dataset from the given file paths.
    """
    dataset = RawCustomDataset(file_paths, config=config)
    return dataset

def evaluate_model(trainer, eval_dataset):
    """
    Perform evaluation and log metrics.
    """
    metrics = trainer.evaluate(eval_dataset=eval_dataset)
    print("Evaluation Metrics:", metrics)

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

if __name__ == "__main__":
    # Load configuration from the YAML file
    with open("../config/eval_config.yaml", "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    # Load the model and tokenizer
    model_dir = config["model"]["model_path"]
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["tokenizer_path"])
    model = T5ForTokenClassification.from_pretrained(model_dir)

    # Collect test dataset files
    test_dataset_folder = config["data"]["test_dataset_folder"]
    src_file_extension = config["data"]["src_file_extension"]
    test_dataset_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(test_dataset_folder)
        for file in files if file.endswith(src_file_extension)
    ]

    # Prepare evaluation dataset
    eval_dataset = load_eval_dataset(test_dataset_files, config)

    # Initialize trainer
    trainer = initialize_trainer(model, tokenizer, config, eval_dataset=eval_dataset)

    # Perform evaluation
    evaluate_model(trainer, eval_dataset)
