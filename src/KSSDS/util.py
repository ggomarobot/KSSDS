import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import TrainerCallback, EvalPrediction
from torch.utils.tensorboard import SummaryWriter
import GPUtil

def initialize_classifier(classifier_layer):
    # Initialize weights using Xavier initialization
    nn.init.xavier_uniform_(classifier_layer.weight)
    
    # Initialize biases to zero
    if classifier_layer.bias is not None:
        nn.init.zeros_(classifier_layer.bias)

# Pre-trained 모델의 인코더와 우리가 만든 인코더의 구조와 weight 값들이 일치하는지 확인
def compare_state_dicts(dict1, dict2):
    # Check if both dictionaries have the same keys
    if dict1.keys() != dict2.keys():
        print("The dictionaries have different keys.")
        return False
    
    # Check if the values for each key are the same
    for key in dict1:
        if not torch.equal(dict1[key], dict2[key]):
            print(f"Values for key '{key}' are different.")
            return False
    
    print("All keys and values match.")
    return True

def compare_state_dicts_after_training(dict1, dict2):
    dict1 = {k: v.cpu() for k, v in dict1.items()}
    dict2 = {k: v.cpu() for k, v in dict2.items()}    
    # Check if both dictionaries have the same keys
    if dict1.keys() != dict2.keys():
        print("The dictionaries have different keys.")
        return False
    
    # Check if the values for each key are the same
    for key in dict1:
        if key not in dict2:
            print(f"Key '{key}' is missing in the second dictionary.")
            return False
        
        if dict1[key].shape != dict2[key].shape:
            print(f"Shape mismatch for key '{key}': {dict1[key].shape} vs {dict2[key].shape}")
            return False
        
        if not torch.allclose(dict1[key], dict2[key], rtol=1e-5, atol=1e-8):
            print(f"Values for key '{key}' are different.")
            return False
    
    print("All keys and values match.")
    return True

def compare_configs_after_training(dict1, dict2):
    differences = {}
    for key in dict1:
        if key not in dict2:
            differences[key] = f'Key {key} not found in second dictionary'
        elif dict1[key] != dict2[key]:
            differences[key] = (dict1[key], dict2[key])
    for key in dict2:
        if key not in dict1:
            differences[key] = f'Key {key} not found in first dictionary'
    return differences

# 텐서보드 콜백 함수 - GPU 사용량 / F1 score, accuracy 등의 metric 기록
class CombinedTensorboardCallback(TrainerCallback):
    def __init__(self):
        self.tb_writer = None
    
    def _init_summary_writer(self, args):
        # Initialize TensorBoard writer
        self.tb_writer = SummaryWriter(log_dir=args.logging_dir)
    
    def _compute_gpu_utilization(self):
        gpus = GPUtil.getGPUs()
        # log the GPU utilization
        avg_gpu_load = sum(gpu.load for gpu in gpus) / len(gpus)
        # log the GPU memory utilization
        avg_mem_util = sum(gpu.memoryUtil for gpu in gpus) / len(gpus)
        return avg_gpu_load, avg_mem_util

    def on_step_end(self, args, state, control, **kwargs):
        # Ensure TensorBoard writer is initialized
        if self.tb_writer is None:
            self._init_summary_writer(args)
        
        # Log GPU utilization
        avg_gpu_compute, avg_gpu_memory = self._compute_gpu_utilization()
        self.tb_writer.add_scalar(
            tag="GPU Utilization / Compute",
            scalar_value=avg_gpu_compute*100,
            global_step=state.global_step
        )
        self.tb_writer.add_scalar(
            tag="GPU Utilization / Memory",
            scalar_value=avg_gpu_memory*100,
            global_step=state.global_step
        )
    
    def on_evaluate(self, args, state, control, **kwargs):
        # Ensure TensorBoard writer is initialized
        if self.tb_writer is None:
            self._init_summary_writer(args)
        # Extract logs from kwargs
        logs = kwargs.get('metrics', {})
        if logs:
            for key, value in logs.items():
                self.tb_writer.add_scalar(
                    tag=f"Evaluation/{key}",
                    scalar_value=value,
                    global_step=state.global_step
                )

def compute_metrics(p: EvalPrediction):
    predictions = p.predictions
    labels = p.label_ids

    # Compute metrics
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')

    return {
        'eval_accuracy': accuracy,
        'eval_precision': precision,
        'eval_recall': recall,
        'eval_f1': f1
    }