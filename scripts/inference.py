import os
import sys
from KSSDS import KSSDS  # PyPI 패키지로 설치된 KSSDS를 가져옴
import yaml

if __name__ == "__main__":
    # config 파일 경로
    config_path = "./config/inference_config.yaml"

    # Initialize KSSDS
    kssds = KSSDS(config_path=config_path)

    # Read inputs from config
    input_tsv = kssds.config.get("input_tsv")
    input_sequence = kssds.config.get("input_sequence")
    output_tsv = kssds.config.get("output_tsv")
    output_print = kssds.config.get("output_print", False)

    # Ensure valid input/output specification
    if (input_tsv and input_sequence) or (not input_tsv and not input_sequence):
        raise ValueError("Specify either 'input_tsv' or 'input_sequence' in the configuration file, but not both.")
    # either an 'output_tsv' file path must be specified in the config file, or set output_print to True      
    if not output_tsv and not output_print:
        raise ValueError("Specify either 'output_tsv' or enable 'output_print'.")  

    # Process inputs based on input type
    if input_tsv:
        kssds.process_tsv(input_tsv, output_tsv, output_print)
    elif input_sequence:
        kssds.process_input_sequence(input_sequence, output_tsv, output_print)