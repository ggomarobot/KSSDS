import os
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import csv

class RawCustomDataset(Dataset):
    def __init__(self, input_data, is_file=True, raw_data_mode=False, max_length=512, src_file_extension=".tsv", config=None):
        if config:
            # Override parameters with values from config if provided
            is_file = config.get("dataset", {}).get("is_file", is_file)
            raw_data_mode = config.get("dataset", {}).get("raw_data_mode", raw_data_mode)
            max_length = config.get("dataset", {}).get("max_length", max_length)
            tokenizer_path = config.get("dataset", {}).get("tokenizer_path", "lcw99/t5-base-korean-text-summary")
            src_file_extension = config.get("dataset", {}).get("src_file_extension", ".tsv")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.raw_data_mode = raw_data_mode
        self.max_length = max_length
        self.is_file = is_file
        self.src_file_extension = src_file_extension
        # training 시 tsv 파일 읽어와서 sequence / label 만듦
        if self.is_file:
            if isinstance(input_data, list):  # If a list of files is provided
                self.split_sentence_list, self.label_list, self.attention_mask_list = self.load_from_multiple_files(input_data)
            elif os.path.isdir(input_data):  # If a folder is provided
                file_list = [os.path.join(input_data, f) for f in os.listdir(input_data) if f.endswith(src_file_extension)]
                self.split_sentence_list, self.label_list, self.attention_mask_list = self.load_from_multiple_files(file_list)
            else:  # If a single file is provided
                self.split_sentence_list, self.label_list, self.attention_mask_list = self.load_from_file(input_data)
        # inference 시 string text 에서 sequence 만듦    
        else:
            self.split_sentence_list, self.attention_mask_list = self.process_long_string(input_data)
            self.label_list = []  # Labels are not needed for inference       
    
    def __len__(self):
        return len(self.split_sentence_list)

    def __getitem__(self, idx):
        item = {
            'input_ids': torch.tensor(self.split_sentence_list[idx], dtype=torch.long),
            'attention_mask': torch.tensor(self.attention_mask_list[idx], dtype=torch.long)
        }
        if self.is_file:  # Labels are only needed for training
            item['labels'] = torch.tensor(self.label_list[idx], dtype=torch.long)
        return item

    def load_from_file(self, file_path):
        return self._load_tsv_data(file_path)

    def load_from_multiple_files(self, file_list):
        split_sentence_list = []
        attention_mask_list = []
        label_list = []

        for file_path in file_list:
            file_sentences, file_labels, file_attention_masks = self._load_tsv_data(file_path)
            split_sentence_list.extend(file_sentences)
            label_list.extend(file_labels)
            attention_mask_list.extend(file_attention_masks)

        return split_sentence_list, label_list, attention_mask_list

    # training 함수 (tsv 파일에서 utterance 로드해오기)
    def _load_tsv_data(self, file_path):
        split_sentence_list = []
        attention_mask_list = []
        label_list = []

        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')

            for row in reader:
                transcription = row['Sentence'].strip()  # Use column name instead of index
                labels_str = row['Labels']  # Use column name instead of index

                tokenized_utterance = self.tokenizer.tokenize(transcription)
                token_ids = [self.tokenizer.convert_tokens_to_ids(t) for t in tokenized_utterance]
                
                transcription_labels = list(map(int, labels_str.split()))
               
                chunks, chunk_labels = self.split_into_chunks_with_boundaries(token_ids, transcription_labels,
                                                                              max_length=self.max_length)
                
                for chunk, chunk_label in zip(chunks, chunk_labels):
                    attention_masks = [1] * len(chunk)

                    split_sentence_list.append(chunk)
                    attention_mask_list.append(attention_masks)
                    label_list.append(chunk_label)

        return split_sentence_list, label_list, attention_mask_list
    
    # inference 시 string 읽어오는 함수
    def process_long_string(self, long_string):
        tokenized_sequence = self.tokenizer.tokenize(long_string)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokenized_sequence)
        
        split_sentence_list = []
        attention_mask_list = []

        chunks = self.split_into_chunks_with_boundaries(token_ids, self.max_length, inference=True)
        
        for tokens in chunks:
            attention_masks = [1] * len(tokens)
            split_sentence_list.append(tokens)
            attention_mask_list.append(attention_masks)
                    
        return split_sentence_list, attention_mask_list
    
    # 목적: input chunk 들을 max length 길이 아래로 나눔
    def split_into_chunks_with_boundaries(self, tokens, labels, max_length=512, inference=False):
        chunks = []
        chunk_labels = []

        # If the token sequence is shorter than the max length, return it as a single chunk
        if len(tokens) <= max_length and not inference:
            chunks.append(tokens)
            chunk_labels.append(labels)
            return chunks, chunk_labels
        elif len(tokens) <= max_length and inference:
            chunks.append(tokens)
            return chunks            

        start = 0
        if inference:
            # Split into chunks of size `max_length` during inference
            while start < len(tokens):
                end = min(start + max_length, len(tokens))  # Ensure end does not exceed token length
                chunk = tokens[start:end]
                chunks.append(chunk)
                start = end
            return chunks
        else:
            while start < len(tokens):
                end = min(start + max_length, len(tokens))  # Ensure end does not exceed token length
                chunk = tokens[start:end]
                chunk_label = labels[start:end]                
                chunks.append(chunk)
                chunk_labels.append(chunk_label)                
                start = end            
            return chunks, chunk_labels