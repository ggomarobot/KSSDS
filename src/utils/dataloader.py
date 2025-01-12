import torch
from torch.utils.data import DataLoader


# Custom collate function
def custom_collate_fn(batch, raw_data_mode=False, inference_mode=False):
    """
    Custom collate function to process batches for the DataLoader.

    Args:
        batch: List of data samples.
        raw_data_mode: Whether to process raw data (human-readable texts) without labels.
        inference_mode: Whether to prepare data for inference.
    
    Returns:
        A dictionary containing tensors for input_ids, attention_mask, and labels (optional).
    """
    if raw_data_mode:
        return {
            'input_ids': torch.tensor([item['input_ids'].tolist() for item in batch], dtype=torch.long),
            'attention_mask': torch.tensor([item['attention_mask'].tolist() for item in batch], dtype=torch.long),
            # Labels might not be present in raw mode
        }
    elif inference_mode:
        # For inference, pad input_ids and attention_masks to the maximum length in the batch
        input_ids = [item['input_ids'].tolist() for item in batch]
        attention_masks = [item['attention_mask'].tolist() for item in batch]

        max_length = max(len(ids) for ids in input_ids)

        padded_input_ids = [ids + [0] * (max_length - len(ids)) for ids in input_ids]
        padded_attention_masks = [mask + [0] * (max_length - len(mask)) for mask in attention_masks]

        return {
            'input_ids': torch.tensor(padded_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(padded_attention_masks, dtype=torch.long),
        }
    else:
        # Standard processing for training and evaluation
        input_ids = [item['input_ids'].tolist() for item in batch]
        attention_masks = [item['attention_mask'].tolist() for item in batch]
        labels = [item['labels'].tolist() for item in batch]

        max_length = max(len(ids) for ids in input_ids)

        padded_input_ids = [ids + [0] * (max_length - len(ids)) for ids in input_ids]
        padded_attention_masks = [mask + [0] * (max_length - len(mask)) for mask in attention_masks]
        padded_labels = [lbl + [-1] * (max_length - len(lbl)) for lbl in labels]

        return {
            'input_ids': torch.tensor(padded_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(padded_attention_masks, dtype=torch.long),
            'labels': torch.tensor(padded_labels, dtype=torch.long)
        }


# Define the custom DataLoader class
class CustomDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0, collate_fn=None,
                 pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None,
                 multiprocessing_context=None, config=None):
        """
        Custom DataLoader to handle configurations and modes.

        Args:
            dataset: The dataset to load.
            batch_size: Number of samples per batch.
            shuffle: Whether to shuffle the data.
            sampler: Sampler for the data.
            batch_sampler: Batch sampler for the data.
            num_workers: Number of worker processes for data loading.
            collate_fn: Custom collate function.
            pin_memory: Whether to use pinned memory.
            drop_last: Whether to drop the last incomplete batch.
            timeout: Timeout for data loading.
            worker_init_fn: Initialization function for workers.
            multiprocessing_context: Multiprocessing context for data loading.
            config: Dictionary or object containing custom configurations.
        """
        # Extract raw_data_mode and inference_mode from the config
        raw_data_mode = config.get("raw_data_mode", False) if config else False
        inference_mode = config.get("inference_mode", False) if config else False

        # Override collate_fn if provided, otherwise use the default
        if collate_fn is None:
            collate_fn = custom_collate_fn
        
        # Pass raw_data_mode to collate_fn
        collate_fn = lambda batch: custom_collate_fn(batch, raw_data_mode=raw_data_mode, inference_mode=inference_mode)

        # Initialize the parent DataLoader class
        super(CustomDataLoader, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            multiprocessing_context=multiprocessing_context
        )
