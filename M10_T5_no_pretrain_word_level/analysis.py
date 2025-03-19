import torch
import argparse
import logging
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler
from transformers import T5ForConditionalGeneration, AutoTokenizer

logger = logging.getLogger(__name__)

class TextDataset:
    def __init__(self, tokenizer, file_path, sample_size=1):
        self.examples = []
        df = pd.read_csv(file_path)
        sources = df["source"].tolist()[:sample_size]  # Use a small sample
        labels = df["target"].tolist()[:sample_size]

        for i in tqdm(range(len(sources))):
            self.examples.append(self.convert_examples_to_features(sources[i], labels[i], tokenizer))

    def convert_examples_to_features(self, source, label, tokenizer):
        source_ids = tokenizer.encode(source, add_special_tokens=True)
        decoder_input_ids = tokenizer.encode(label, add_special_tokens=True)

        source_ids = source_ids[:512] + [tokenizer.pad_token_id] * (512 - len(source_ids))
        decoder_input_ids = decoder_input_ids[:256] + [tokenizer.pad_token_id] * (256 - len(decoder_input_ids))

        return {
            "input_ids": torch.tensor(source_ids),
            "decoder_input_ids": torch.tensor(decoder_input_ids),
            "source_text": source,
            "target_text": label
        }

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        example = self.examples[i]
        return example["input_ids"], example["decoder_input_ids"], example["source_text"], example["target_text"]


def evaluate(model, tokenizer, dataset):
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=1)

    model.eval()
    results = []

    for batch in tqdm(dataloader, total=len(dataloader)):
        input_ids, decoder_input_ids, source_text, target_text = batch
        input_ids, decoder_input_ids = input_ids.squeeze(1), decoder_input_ids.squeeze(1)

        with torch.no_grad():
            outputs = model.generate(input_ids=input_ids.to(model.device), num_beams=5, max_length=256, return_dict_in_generate=True, output_scores=True)

        beam_candidates = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs.sequences]
        best_prediction = beam_candidates[0]

        embeddings = model.get_input_embeddings()(input_ids.to(model.device)).detach().cpu().numpy()
        last_10_embeddings = embeddings[-10:]

        tokenized_input_ids = input_ids.tolist()
        tokenized_ground_truth = decoder_input_ids.tolist()

        correct = 1 if best_prediction.strip() == target_text.strip() else 0

        results.append({
            "Raw Input": source_text,
            "Tokenized Input IDs": tokenized_input_ids,
            "Last 10 Embeddings": last_10_embeddings.tolist(),
            "Model Output Before Beam Search": tokenizer.decode(outputs.sequences[0], skip_special_tokens=True),
            "Beam Candidates": beam_candidates,
            "Ground Truth": target_text,
            "Tokenized Ground Truth": tokenized_ground_truth,
            "Accuracy": correct
        })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--model_name", type=str, required=True, help="Model name")
    parser.add_argument("--config_name", type=str, required=True, help="Config name")
    parser.add_argument("--test_data_file", type=str, required=True, help="Test dataset file")
    parser.add_argument("--encoder_block_size", type=int, default=512, help="Max input length")
    parser.add_argument("--decoder_block_size", type=int, default=256, help="Max output length")
    parser.add_argument("--eval_batch_size", type=int, default=1, help="Batch size for evaluation")
    parser.add_argument("--do_test", action='store_true', help="Run test evaluation")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer correctly
    tokenizer = AutoTokenizer.from_pretrained(args.config_name)

    # Load model
    model_path = os.path.join(args.output_dir, args.model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)

    dataset = TextDataset(tokenizer, args.test_data_file, sample_size=args.eval_batch_size)
    
    if args.do_test:
        results = evaluate(model, tokenizer, dataset)
    
        for res in results:
            print("\n===== Sample Output =====")
            for key, value in res.items():
                print(f"{key}: {value}")

if __name__ == "__main__":
    main()
