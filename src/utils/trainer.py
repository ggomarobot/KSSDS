import numpy as np
import torch
from transformers import Trainer, EvalPrediction
from .dataloader import CustomDataLoader

class CustomTrainer(Trainer):
    def __init__(self, *args, custom_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_config = custom_config  # Store the custom config for later use    
    
    def get_eval_dataloader(self, eval_dataset=None):
        if eval_dataset is None:
            eval_dataset = self.eval_dataset

        shuffle_eval = self.custom_config.get("shuffle_eval", False)
        
        return CustomDataLoader(
            eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            num_workers=self.args.dataloader_num_workers,
            shuffle=shuffle_eval,  # Use the custom shuffle_eval argument
            config=self.custom_config
        )

    def evaluate(self, eval_dataset=None, ignore_keys=None):
        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        # Initialize containers
        all_preds = []
        all_labels = []

        self.model.eval()

        for batch in eval_dataloader:
            # Use inference mode (instead of torch.no_grad) for efficiency
            with torch.inference_mode():
                # Move inputs to device
                inputs = {
                    k: v.to(self.args.device)
                    for k, v in batch.items()
                    if k in ['input_ids', 'attention_mask', 'labels']
                }

                # Forward pass
                outputs = self.model(**inputs)
                logits = outputs[1]

                # Collect predictions and labels
                preds = logits.argmax(dim=-1).cpu().numpy() if logits.ndim > 1 else logits.cpu().numpy()
                labels = inputs['labels'].cpu().numpy()

                all_preds.append(preds)
                all_labels.append(labels)

        # Flatten all predictions and labels
        flattened_preds = []
        flattened_labels = []
        for preds, labels in zip(all_preds, all_labels):
            flattened_preds.extend(preds.reshape(-1))
            flattened_labels.extend(labels.reshape(-1))

        flattened_preds = np.array(flattened_preds)
        flattened_labels = np.array(flattened_labels)

        # Create valid mask based on label values
        valid_mask = flattened_labels != -1

        if len(flattened_preds) != len(valid_mask):
            raise ValueError("Predictions and valid_mask length mismatch.")

        # Filter out invalid entries
        filtered_labels = flattened_labels[valid_mask]
        filtered_preds = flattened_preds[valid_mask]

        # Calculate metrics
        metrics = self.compute_metrics(EvalPrediction(predictions=filtered_preds, label_ids=filtered_labels))

        # Log metrics to TensorBoard
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        return metrics