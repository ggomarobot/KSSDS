import yaml
import torch
import csv
import transformers
from transformers import AutoTokenizer
from .T5_encoder import T5ForTokenClassification
from .dataloader import CustomDataLoader, custom_collate_fn
from .dataset import RawCustomDataset


class KSSDS:
    def __init__(self, config_path=None, model_path=None, tokenizer_path=None, max_repeats=60, detection_threshold=70):
        if config_path:
            # Load configuration from YAML file
            with open(config_path, 'r') as file:
                self.config = yaml.safe_load(file)
        else:
            transformers.logging.set_verbosity_error()
            # Use default or user-provided settings
            self.config = {
                "model_path": model_path or "ggomarobot/KSSDS",
                "tokenizer_path": tokenizer_path or "ggomarobot/KSSDS",
                "repetition_detection": {
                    "max_repeats": max_repeats,
                    "detection_threshold": detection_threshold
                },
                "batch_size": 1,  # Default batch size
                "inference_mode": True
            }

        # Load model and tokenizer
        self.model_dir = self.config["model_path"]
        self.model, self.device = self.load_model()
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["tokenizer_path"])
        self.max_repeats = self.config["repetition_detection"]["max_repeats"]
        self.detection_threshold = self.config["repetition_detection"]["detection_threshold"]

    def load_model(self):
        model = T5ForTokenClassification.from_pretrained(self.model_dir)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        return model, device

    def prepare_input_data(self, text):
        dataset = RawCustomDataset(input_data=text, is_file=False, config=self.config)
        dataloader = CustomDataLoader(dataset, batch_size=self.config["batch_size"],
                                      collate_fn=custom_collate_fn, config=self.config)
        return dataloader

    def trim_predictions(self, input_ids, predictions, attention_masks):
        trimmed_inputs = []
        trimmed_predictions = []
        
        for inp_id, pred, mask in zip(input_ids, predictions, attention_masks):
            if len(pred) != len(mask):
                print(f"Warning: Length mismatch between pred and mask: {len(pred)} vs {len(mask)}")
                min_length = min(len(pred), len(mask))
                pred = pred[:min_length]
                mask = mask[:min_length]
                inp_id = inp_id[:min_length]

            trimmed_pred = pred[mask == 1]
            trimmed_inp_id = inp_id[mask == 1]

            trimmed_predictions.append(trimmed_pred)
            trimmed_inputs.append(trimmed_inp_id)

        return trimmed_inputs, trimmed_predictions

    def segment_predictions(self, inp, pred):
        segments = []
        current_segment = []
        inp = inp[0].cpu().numpy()
        pred = pred[0]

        for token, label in zip(inp, pred):
            if label == 1:
                if current_segment:
                    current_segment.append(token)
                    segments.append(current_segment)
                    current_segment = []
                else:
                    segments.append([token])
            else:
                current_segment.append(token)

        if current_segment:
            segments.append(current_segment)

        return segments
    
    def handle_repetitions(self, decoded_preds, max_repeats=60, detection_threshold=70):
        words = decoded_preds.split()
        if len(words) <= detection_threshold:
            return [decoded_preds]

        result_sentences = []
        current_sentence = []
        current_repetition = []

        def flush_repetition():
            if len(current_repetition) > max_repeats:
                for j in range(0, len(current_repetition), max_repeats):
                    result_sentences.append(" ".join(current_repetition[j:j + max_repeats]))
            else:
                current_sentence.extend(current_repetition)

        for i, word in enumerate(words):
            if i > 0 and word == words[i - 1]:
                if current_sentence:
                    last_word = current_sentence.pop()
                    if current_sentence:
                        result_sentences.append(" ".join(current_sentence))
                    current_sentence = []
                    current_repetition.append(last_word)
                current_repetition.append(word)
            else:
                if current_repetition:
                    flush_repetition()
                    current_repetition = []
                current_sentence.append(word)

        if current_repetition:
            flush_repetition()

        if current_sentence:
            result_sentences.append(" ".join(current_sentence))

        return result_sentences

    
    def process_predictions(self, input_ids, predictions, attention_masks):
        results = []
        carry_over_tokens = []  # Tokens to carry over to the next batch
        carry_over_labels = []  # Corresponding labels for carry-over tokens

        for inp_id, pred, mask in zip(input_ids, predictions, attention_masks):
            # Convert inputs to tensors if they are not already
            inp_id = torch.tensor(inp_id, device=self.device) if not isinstance(inp_id, torch.Tensor) else inp_id
            pred = torch.tensor(pred, device=self.device) if not isinstance(pred, torch.Tensor) else pred
            mask = torch.tensor(mask, device=self.device) if not isinstance(mask, torch.Tensor) else mask

            # If there's a carry-over, prepend it to the current batch
            if carry_over_tokens:
                carry_over_tokens_tensor = torch.tensor(carry_over_tokens, device=self.device)
                carry_over_labels_tensor = torch.tensor(carry_over_labels, device=self.device)
                inp_id = torch.cat((carry_over_tokens_tensor, inp_id))
                pred = torch.cat((carry_over_labels_tensor, pred))
                mask = torch.cat((torch.ones(len(carry_over_tokens), device=self.device), mask))
                carry_over_tokens = []
                carry_over_labels = []

            # Process the current batch
            trimmed_inp, trimmed_pred = self.trim_predictions([inp_id], [pred], [mask])
            segmented_preds = self.segment_predictions(trimmed_inp, trimmed_pred)

            # Handle segments and check for carry-over
            for i, seg in enumerate(segmented_preds):
                if i == len(segmented_preds) - 1 and seg[-1] != 1:  # Last segment, and it ends in 0
                    carry_over_tokens = seg  # Carry over this segment
                    carry_over_labels = [0] * len(seg)  # Assign Label: 0 to all carried-over tokens
                else:
                    # Decode and add non-empty results
                    decoded_preds = self.tokenizer.decode(seg, skip_special_tokens=False, clean_up_tokenization_spaces=False).strip()
                    # Handle repetitions
                    decoded_preds = self.handle_repetitions(decoded_preds, self.max_repeats, self.detection_threshold)
                    if decoded_preds:
                        results.extend(decoded_preds)

        # Handle any remaining carry-over at the end of all batches
        if carry_over_tokens:
            decoded_preds = self.tokenizer.decode(carry_over_tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False).strip()
            decoded_preds = self.handle_repetitions(decoded_preds, self.max_repeats, self.detection_threshold)
            if decoded_preds:
                results.extend(decoded_preds)

        return results
    
    def run_inference(self, dataloader):
        self.model.eval()
        all_input_ids = []
        all_predictions = []
        all_attention_masks = []

        for batch in dataloader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch.get('attention_mask', torch.ones_like(input_ids)).to(self.device)

            with torch.inference_mode():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1).cpu().numpy()
            attention_masks = attention_mask.cpu().numpy()

            trimmed_tokens, trimmed_predictions = self.trim_predictions(input_ids, predictions, attention_masks)
            all_input_ids.extend(trimmed_tokens)
            all_predictions.extend(trimmed_predictions)
            all_attention_masks.extend(attention_masks)

        return self.process_predictions(all_input_ids, all_predictions, all_attention_masks)

    def sentence_splitter(self, input_sequence):
        dataloader = self.prepare_input_data(input_sequence)
        return self.run_inference(dataloader)

    def process_tsv(self, input_tsv, output_tsv=None, output_print=False):
        # Get input column names from the configuration
        input_columns = self.config.get("input_columns", {})
        file_path_column = input_columns.get("file_path", "File Path")  # Default: "File Path"
        transcription_column = input_columns.get("transcription", "Transcription")  # Default: "Transcription"

        # Open the input TSV file
        with open(input_tsv, 'r', encoding='utf-8') as infile:
            reader = csv.DictReader(infile, delimiter='\t')
            fieldnames = ['File Name', 'Index', 'Sentence']  # Fixed output column names

            results = []

            # Process each row in the input TSV
            for row in reader:
                file_name = row.get(file_path_column, "").strip()  # Get file path
                transcription = row.get(transcription_column, "").strip()  # Get transcription

                # Split sentences using the KSSDS model
                split_sentences = self.sentence_splitter(transcription)

                for idx, sentence in enumerate(split_sentences):
                    results.append((file_name, idx, sentence.strip()))

            # Write results to output TSV if specified
            if output_tsv:
                with open(output_tsv, 'w', encoding='utf-8', newline='') as outfile:
                    writer = csv.writer(outfile, delimiter='\t')
                    writer.writerow(fieldnames)
                    writer.writerows(results)

            # Print results to terminal if specified
            if output_print:
                for file_name, idx, sentence in results:
                    print(f"{file_name}\t{idx}\t{sentence}")

    def process_input_sequence(self, input_sequence, output_tsv=None, output_print=False):
        split_sentences = self.sentence_splitter(input_sequence)
        
        if output_tsv:
            with open(output_tsv, 'w', encoding='utf-8', newline='') as outfile:
                writer = csv.writer(outfile, delimiter='\t')
                writer.writerow(['Index', 'Sentence'])
                for idx, sentence in enumerate(split_sentences):
                    writer.writerow([idx, sentence.strip()])

        if output_print:
            for idx, sentence in enumerate(split_sentences):
                print(f"[{idx}]: {sentence.strip()}")

    # function for KSSDS package; performs same task as sentence_splitter()            
    def split_sentences(self, input_sequence):
        """
        Split a single string input into sentences using the model.
        Args:
            input_sequence (str): The input text to be split.
        Returns:
            List[str]: A list of split sentences.
        """
        dataloader = self.prepare_input_data(input_sequence)
        return self.run_inference(dataloader)                

if __name__ == "__main__":
    transformers.logging.set_verbosity_error()
    config_path = "./config/inference_config.yaml"

    # Initialize KSSDS
    kssds = KSSDS(config_path=config_path)

    # Read inputs from config
    input_tsv = kssds.config.get("input_tsv")
    input_sequence = kssds.config.get("input_sequence")
    output_tsv = kssds.config["output_tsv"]
    output_print = kssds.config.get("output_print", False)
    
    # Ensure valid input/output specification
    if (input_tsv and input_sequence) or (not input_tsv and not input_sequence):
        raise ValueError("You must specify either 'input_tsv' or input_sequence' in the configuration file, but not both.")
    # either an 'output_tsv' file path must be specified in the config file, or set output_print to True  
    if not output_tsv and not output_print:
        raise ValueError("You must specify either 'output_tsv' or enable 'output_print'.")       

    # Process inputs based on input type
    if input_tsv:
        kssds.process_tsv(input_tsv, output_tsv, output_print)
    elif input_sequence:
        kssds.process_input_sequence(input_sequence, output_tsv, output_print)
