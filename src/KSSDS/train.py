import os
from transformers import AutoTokenizer, TrainingArguments, AutoModelForSeq2SeqLM, EarlyStoppingCallback
from .dataset import RawCustomDataset
from .dataloader import custom_collate_fn
from .T5_encoder import T5ForTokenClassification
from .util import initialize_classifier, compute_metrics, CombinedTensorboardCallback
from .trainer import CustomTrainer

def train_model(training_dataset_files, validation_dataset_files, config):
    """
    Trains the model using the provided training and validation datasets and configuration.
    """
    model_path = config["model"]["model_path"]
    num_labels = config["model"]["num_labels"]
    '''
    # 첫 훈련 시 코드 #
    ###########################################################################
    # 기존 토크나이저
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # 우리 모델
    model = T5ForTokenClassification.from_pretrained(model_path, num_labels=num_labels) 
    # classifier.weight / classifier.bias initialization
    initialize_classifier(model.classifier)    
    ###########################################################################
    '''
    #'''
    # 첫 훈련 이후 코드 #
    ###########################################################################
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = T5ForTokenClassification.from_pretrained(model_path)
    ###########################################################################
    #'''
    # Create dataset objects
    train_dataset = RawCustomDataset(input_data=training_dataset_files, config=config)
    eval_dataset = RawCustomDataset(input_data=validation_dataset_files, config=config)

    # Check early stopping settings in the YAML config
    early_stopping_config = config["training"].get("early_stopping", {})
    callbacks = [CombinedTensorboardCallback]

    if early_stopping_config.get("enabled", False):
        patience = early_stopping_config.get("patience", 2)  # Default patience of 2
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))


    # Load training arguments from the configuration file
    training_args = TrainingArguments(
        output_dir=config["training"]["output_dir"],
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=config["training"]["per_device_eval_batch_size"],
        num_train_epochs=config["training"]["num_train_epochs"],
        dataloader_num_workers=config["training"]["dataloader_num_workers"],
        logging_dir=config["training"]["logging_dir"],
        eval_strategy=config["training"]["eval_strategy"],
        save_total_limit=config["training"]["save_total_limit"],
        remove_unused_columns=config["training"]["remove_unused_columns"],
        logging_strategy=config["training"]["logging_strategy"],
        save_strategy=config["training"]["save_strategy"],
        save_safetensors=config["training"]["save_safetensors"],
        report_to=config["training"]["report_to"],
        load_best_model_at_end=config["training"]["load_best_model_at_end"],
        metric_for_best_model=config["training"]["metric_for_best_model"]
    )

    # Initialize the trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=custom_collate_fn,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        custom_config=config["dataloader"]
    )
    # Check if there are existing checkpoints to resume training from
    checkpoints = [
        d for d in os.listdir(training_args.output_dir)
        if d.startswith('checkpoint-') and os.path.isdir(os.path.join(training_args.output_dir, d))
    ]
    if checkpoints:
        # Sort checkpoints to find the most recent one
        checkpoints.sort(key=lambda x: int(x.split('-')[1]), reverse=True)
        latest_checkpoint = os.path.join(training_args.output_dir, checkpoints[0])
        print(f"Loading from checkpoint: {latest_checkpoint}")
        trainer.train(resume_from_checkpoint=latest_checkpoint)
    else:
        print("Starting a new training")
        trainer.train()

    # Save the final model
    model.save_pretrained(training_args.output_dir,
                          safe_serialization=config["training"]["safe_serialization"])
    tokenizer.save_pretrained(training_args.output_dir)
    
    return trainer  # Return the trainer object for further use
