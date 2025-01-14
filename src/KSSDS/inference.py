import yaml
import torch
import csv
import transformers
from transformers import AutoTokenizer
from .T5_encoder import T5ForTokenClassification


class KSSDS:
    def __init__(self, config_path=None, model_path=None, tokenizer_path=None, max_repeats=60, detection_threshold=70, max_phrase_length=2):
        transformers.logging.set_verbosity_error()
        if config_path:
            with open(config_path, 'r') as file:
                self.config = yaml.safe_load(file)
        else:
            self.config = {
                "model_path": model_path or "ggomarobot/KSSDS",
                "tokenizer_path": tokenizer_path or "ggomarobot/KSSDS",
                "repetition_detection": {
                    "max_repeats": max_repeats,
                    "detection_threshold": detection_threshold,
                    "max_phrase_length": max_phrase_length
                },
                "max_length": 512,
            }

        self.model, self.device = self.load_model(self.config["model_path"])
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["tokenizer_path"])
        self.max_length = self.config["max_length"]
        self.max_repeats = self.config["repetition_detection"]["max_repeats"]
        self.detection_threshold = self.config["repetition_detection"]["detection_threshold"]
        self.max_phrase_length = self.config["repetition_detection"]["max_phrase_length"]

    def load_model(self, model_path):
        model = T5ForTokenClassification.from_pretrained(model_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        return model, device

    def split_into_chunks(self, tokens):
        return [tokens[i:i + self.max_length] for i in range(0, len(tokens), self.max_length)]

    def process_text(self, text):
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        chunks = self.split_into_chunks(tokens)
        return chunks
    '''
    def handle_repetitions(self, text):
        words = text.split()
        if len(words) <= self.detection_threshold:
            return [text]

        result_sentences = []
        current_repetition = []
        current_sentence = []

        def flush_repetition():
            if len(current_repetition) > self.max_repeats:
                result_sentences.extend(
                    [" ".join(current_repetition[i:i + self.max_repeats]) for i in range(0, len(current_repetition), self.max_repeats)]
                )
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
    '''
    def handle_repetitions(self, text):
        """
        Handles single-word and phrase repetitions in the text, ensuring proper order and separation.

        Args:
            text (str): The input text to process.

        Returns:
            List[str]: The processed text split into sentences.
        """
        words = text.split()
        if len(words) <= self.detection_threshold:
            return [text]

        result_sentences = []
        current_repetition = []
        current_sentence = []

        def flush_sentence():
            """Flush the current sentence into result_sentences."""
            if current_sentence:
                result_sentences.append(" ".join(current_sentence))
                current_sentence.clear()

        def flush_repetition():
            """Flush the current repetition into result_sentences."""
            for i in range(0, len(current_repetition), phrase_length * self.max_repeats):
                chunk = current_repetition[i:i + phrase_length * self.max_repeats]
                result_sentences.append(" ".join(chunk))
            current_repetition.clear()

        def find_repeating_phrase(start_idx):
            """Find the smallest repeating phrase starting at the given index."""
            for phrase_length in range(1, self.max_phrase_length + 1):
                phrase = words[start_idx:start_idx + phrase_length]
                next_idx = start_idx + phrase_length
                if next_idx + phrase_length <= len(words) and words[next_idx:next_idx + phrase_length] == phrase:
                    return phrase
            return None

        i = 0
        while i < len(words):
            repeating_phrase = find_repeating_phrase(i)
            if repeating_phrase:
                # Flush any ongoing sentence before handling repetition
                flush_sentence()

                # Accumulate repeating phrases
                phrase_length = len(repeating_phrase)
                while i + phrase_length <= len(words) and words[i:i + phrase_length] == repeating_phrase:
                    current_repetition.extend(repeating_phrase)
                    i += phrase_length

                # Flush accumulated repetition if it reaches the threshold
                if len(current_repetition) >= phrase_length * self.max_repeats:
                    flush_repetition()
            else:
                # Add non-repeating words to the current sentence
                if current_repetition:
                    # Flush repetition before starting a new sentence
                    flush_repetition()
                current_sentence.append(words[i])
                i += 1

        # Flush any remaining tokens
        flush_sentence()
        flush_repetition()

        return result_sentences
        
    def segment_predictions(self, inp, pred):
        segments = []
        current_segment = []
        inp = inp[0]
        pred = pred[0]

        for token, label in zip(inp, pred):
            if label == 1:  # End of a sentence
                if current_segment:
                    current_segment.append(token)
                    segments.append(current_segment)
                    current_segment = []
                else:
                    segments.append([token])
            else:  # Continuation of a sentence
                current_segment.append(token)

        if current_segment:  # Add any remaining tokens as a final segment
            segments.append(current_segment)

        return segments


    def run_inference(self, input_sequence):
        chunks = self.process_text(input_sequence)
        results = []
        carry_over_tokens = []  # Tokens to carry over to the next chunk
        carry_over_labels = []  # Corresponding labels for carry-over tokens

        self.model.eval()
        with torch.inference_mode():
            for chunk in chunks:
                # Prepare inputs
                input_ids = torch.tensor([chunk]).to(self.device)
                attention_mask = torch.ones_like(input_ids).to(self.device)

                # Model inference
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=-1).squeeze().tolist()
                # Ensure predictions is a list, even for single-token inputs
                if isinstance(predictions, int):
                    predictions = [predictions]
                # Handle carry-over tokens from previous chunk
                if carry_over_tokens:
                    chunk = carry_over_tokens + chunk
                    predictions = carry_over_labels + predictions
                    carry_over_tokens = []
                    carry_over_labels = []
                # Segment predictions into sentences
                segmented_predictions = self.segment_predictions([chunk], [predictions])

                # Process each segment
                for i, segment in enumerate(segmented_predictions):
                    if i == len(segmented_predictions) - 1 and segment[-1] != 1:  # Last segment does not end in a sentence
                        carry_over_tokens = segment  # Carry over this segment
                        carry_over_labels = [0] * len(segment)  # Assign label 0 for carry-over tokens
                    else:
                        decoded_sentence = self.tokenizer.decode(segment, skip_special_tokens=False, clean_up_tokenization_spaces=False).strip()
                        # Handle repetitions
                        decoded_sentence = self.handle_repetitions(decoded_sentence)
                        if decoded_sentence:
                            results.extend(decoded_sentence)

        # Handle any remaining carry-over tokens
        if carry_over_tokens:
            decoded_remainder = self.tokenizer.decode(carry_over_tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False).strip()
            decoded_remainder = self.handle_repetitions(decoded_remainder)
            if decoded_remainder:
                results.extend(decoded_remainder)

        return results

    # Wrapper function for KSSDS package; performs sentence splitting
    def split_sentences(self, input_sequence):
        """
        Split a single string input into sentences using the model.
        Args:
            input_sequence (str): The input text to be split.
        Returns:
            List[str]: A list of split sentences.
        """
        return self.run_inference(input_sequence)

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
                split_sentences = self.run_inference(transcription)

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
        split_sentences = self.run_inference(input_sequence)

        if output_tsv:
            with open(output_tsv, 'w', encoding='utf-8', newline='') as outfile:
                writer = csv.writer(outfile, delimiter='\t')
                writer.writerow(['Index', 'Sentence'])
                for idx, sentence in enumerate(split_sentences):
                    writer.writerow([idx, sentence.strip()])

        if output_print:
            for idx, sentence in enumerate(split_sentences):
                print(f"[{idx}]: {sentence.strip()}")