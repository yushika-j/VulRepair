# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import absolute_import, division, print_function
import argparse
import logging
import os
import random
import numpy as np
import torch
import json
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from transformers import (AdamW, get_linear_schedule_with_warmup, 
                          T5ForConditionalGeneration, T5Config, T5Tokenizer)
from tqdm import tqdm
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from tokenizers import Tokenizer

cpu_cont = 16
logger = logging.getLogger(__name__)

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_ids,
                 label,
                 decoder_input_ids):
        self.input_ids = input_ids
        self.label=label
        self.decoder_input_ids = decoder_input_ids
        

class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_type="train"):
        if file_type == "train":
            file_path = args.train_data_file
        elif file_type == "eval":
            file_path = args.eval_data_file
        elif file_type == "test":
            file_path = args.test_data_file
        self.examples = []
        df = pd.read_csv(file_path)
        sources = df["source"].tolist()
        labels = df["target"].tolist()
        for i in tqdm(range(len(sources))):
            self.examples.append(convert_examples_to_features(sources[i], labels[i], tokenizer, args))
        if file_type == "train":
            for example in self.examples[:3]:
                    logger.info("*** Example ***")
                    logger.info("label: {}".format(example.label))
                    logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))
                    logger.info("decoder_input_ids: {}".format(' '.join(map(str, example.decoder_input_ids))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):       
        return self.examples[i].input_ids, self.examples[i].input_ids.ne(0), self.examples[i].label, self.examples[i].decoder_input_ids


def convert_examples_to_features(source, label, tokenizer, args):
    # encode
    source_ids = tokenizer.encode(source)
    source_ids = source_ids.ids
    if len(source_ids) > 510:
        source_ids = source_ids[:510]
        source_ids = [1] + source_ids + [2]
    elif len(source_ids) < 510:
        padding = 510 - len(source_ids)
        source_ids = [1] + source_ids + [2]
        for _ in range(padding):
            source_ids.append(0)
    elif len(source_ids) == 510:
        source_ids = [1] + source_ids + [2]
    source_ids = torch.tensor(source_ids)
    
    decoder_input_ids = tokenizer.encode(label)
    decoder_input_ids = decoder_input_ids.ids
    if len(decoder_input_ids) > 254:
        decoder_input_ids = decoder_input_ids[:254]
        decoder_input_ids = [1] + decoder_input_ids + [2]
    elif len(decoder_input_ids) < 254:
        padding = 254 - len(decoder_input_ids)
        decoder_input_ids = [1] + decoder_input_ids + [2]
        for _ in range(padding):
            decoder_input_ids.append(0)
    elif len(decoder_input_ids) == 254:
        decoder_input_ids = [1] + decoder_input_ids + [2]

    assert len(decoder_input_ids) == 256 and len(source_ids) == 512
    decoder_input_ids = torch.tensor(decoder_input_ids)
    label = decoder_input_ids
    return InputFeatures(source_ids, label, decoder_input_ids)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def train(args, train_dataset, model, tokenizer, eval_dataset):
    """ Train the model """
    # build dataloader
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=0)
    
    args.max_steps = args.epochs * len(train_dataloader)

    # evaluate model per epoch
    args.save_steps = len(train_dataloader) * 1
   
    args.warmup_steps = args.max_steps // 5
    model.to(args.device)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)
    
    # multi-gpu training
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size//max(args.n_gpu, 1))
    logger.info("  Total train batch size = %d",args.train_batch_size*args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)
    
    global_step = 0
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_loss = 100

    writer_path = "tb/codet5_training_loss"
    writer = SummaryWriter(writer_path)

    model.zero_grad()

    for idx in range(args.epochs): 
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        tr_num = 0
        train_loss = 0
        for step, batch in enumerate(bar):
            (input_ids, attention_mask, labels, decoder_input_ids) = [x.squeeze(1).to(args.device) for x in batch]
            model.train()
            # the forward function automatically creates the correct decoder_input_ids
            loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            tr_loss += loss.item()
            tr_num += 1
            train_loss += loss.item()
            if avg_loss == 0:
                avg_loss = tr_loss
            avg_loss = round(train_loss/tr_num,5)
            bar.set_description("epoch {} loss {}".format(idx,avg_loss))
            
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()  
                global_step += 1
                output_flag = True
                avg_loss = round(np.exp((tr_loss - logging_loss) /(global_step- tr_nb)),4)
                if global_step % args.save_steps == 0:
                    # placeholder of evaluation
                    eval_loss = evaluate(args, model, tokenizer, eval_dataset, eval_when_training=True)    
                    # Save model checkpoint
                    if eval_loss < best_loss:
                        best_loss = eval_loss
                        logger.info("  "+"*"*20)  
                        logger.info("  Best Loss:%s",round(best_loss,4))
                        logger.info("  "+"*"*20)                          
                        checkpoint_prefix = 'checkpoint-best-loss'
                        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)                        
                        model_to_save = model.module if hasattr(model,'module') else model
                        output_dir = os.path.join(output_dir, '{}'.format(args.model_name)) 
                        torch.save(model_to_save.state_dict(), output_dir)
                        logger.info("Saving model checkpoint to %s", output_dir)

def clean_tokens(tokens):
    tokens = tokens.replace("[PAD]", "")
    tokens = tokens.replace("[CLS]", "")
    tokens = tokens.replace("[SEP]", "")
    tokens = tokens.strip("\n")
    tokens = tokens.strip()
    return tokens

def evaluate(args, model, tokenizer, eval_dataset, eval_when_training=False):
    #build dataloader
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=0)
    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)
    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    model.eval()
    
    eval_loss, num = 0, 0
    bar = tqdm(eval_dataloader, total=len(eval_dataloader))
    for batch in bar:
        (input_ids, attention_mask, labels, decoder_input_ids) = [x.squeeze(1).to(args.device) for x in batch]
        loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
        if args.n_gpu > 1:
            loss = loss.mean()
        eval_loss += loss.item()
        num += 1
    eval_loss = round(eval_loss/num,5)
    model.train()
    logger.info("***** Eval results *****")
    logger.info(f"Evaluation Loss: {str(eval_loss)}")
    return eval_loss

def test(args, model, tokenizer, test_dataset, best_threshold=0.5):
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=1, num_workers=0)  # Batch size 1 for analysis

    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()

    selected_example = None  # Store one example for debugging

    bar = tqdm(test_dataloader, total=len(test_dataloader))
    for batch in bar:
        correct_pred = False
        (input_ids, attention_mask, labels, decoder_input_ids) = [x.squeeze(1).to(args.device) for x in batch]

        # Step 1: **Raw Input Code**
        raw_input_text = tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)
        raw_ground_truth = tokenizer.decode(decoder_input_ids[0].tolist(), skip_special_tokens=True)

        logger.info("\n==== 1️⃣ Raw Input Code ====")
        logger.info(raw_input_text)
        logger.info("\n==== 2️⃣ Raw Ground Truth ====")
        logger.info(raw_ground_truth)

        # Step 2: **Tokenized Input**
        tokenized_input = input_ids[0].tolist()
        logger.info("\n==== 3️⃣ Tokenized Representation ====")
        logger.info(f"Tokenized Input IDs: {tokenized_input}")

        # Step 3: **Embedding Vectors**
        with torch.no_grad():
            token_embeddings = model.get_input_embeddings()(input_ids)  # Extract embeddings

        logger.info("\n==== 4️⃣ Embedding Vectors ====")
        logger.info(f"Shape: {token_embeddings.shape}")  # (batch_size, seq_length, embedding_dim)
        for idx, token_id in enumerate(input_ids[0][:10]):  # Show first 10 tokens
            token = tokenizer.convert_ids_to_tokens(token_id.item())  
            embedding_vector = token_embeddings[0, idx].tolist()[:5]  # Show first 5 values of the vector
            logger.info(f"Token: {token} | Embedding: {embedding_vector}...")

        # Step 4: **Transformer Model Processing**
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids)
            logits = outputs.logits
            greedy_output = torch.argmax(logits, dim=-1)

        greedy_decoded = tokenizer.decode(greedy_output[0].tolist(), skip_special_tokens=True)
        logger.info("\n==== 5️⃣ Model Output Before Beam Search (Greedy Decoding) ====")
        logger.info(greedy_decoded)

        # Step 5: **Beam Search**
        with torch.no_grad():
            beam_outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,  
                num_beams=args.num_beams,
                num_return_sequences=args.num_beams,  
                max_length=args.decoder_block_size
            )

        beam_outputs = beam_outputs.detach().cpu().tolist()

        # Log beam search candidates
        logger.info("\n==== 6️⃣ Beam Search Candidates ====")
        best_prediction = None
        for idx, single_output in enumerate(beam_outputs):
            prediction = tokenizer.decode(single_output, skip_special_tokens=True)
            logger.info(f"Beam {idx + 1}: {prediction}")

            # Check if any beam matches ground truth
            if prediction == raw_ground_truth:
                best_prediction = prediction
                correct_pred = True
                break
        
        # Step 6: **Final Output vs. Ground Truth**
        logger.info("\n==== 7️⃣ Final Output vs. Ground Truth ====")
        if best_prediction:
            logger.info(f"✅ Correct Prediction Found! Best Beam: {best_prediction}")
        else:
            logger.info(f"❌ No Correct Beam Found. Closest Prediction: {tokenizer.decode(beam_outputs[0], skip_special_tokens=True)}")

        # Save one example for report
        if selected_example is None:
            selected_example = {
                "raw_input": raw_input_text,
                "tokenized_input": input_ids[0].tolist(),
                "embedding_vectors": token_embeddings[0].tolist(), 
                "beam_candidates": [tokenizer.decode(b, skip_special_tokens=True) for b in beam_outputs],
                "ground_truth": raw_ground_truth,
                "selected_output": best_prediction if best_prediction else tokenizer.decode(beam_outputs[0], skip_special_tokens=True),
            }

        break  # Stop after one example for debugging

    # Save the selected example for visualization
    if selected_example:
        with open("model10_example_analysis.json", "w") as f:
            json.dump(selected_example, f, indent=4)

    logger.info("Saved example analysis in `model10_example_analysis.json`")


def main():
    parser = argparse.ArgumentParser()
    # Params
    parser.add_argument("--train_data_file", default=None, type=str, required=False,
                        help="The input training data file (a csv file).")
    parser.add_argument("--output_dir", default=None, type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--model_type", default="t5", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--encoder_block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--decoder_block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--num_beams", default=50, type=int,
                        help="Beam size to use when decoding.")                          
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--model_name", default="model.bin", type=str,
                        help="Saved model name.")
    parser.add_argument("--checkpoint_model_name", default="non_domain_model.bin", type=str,
                            help="Checkpoint model name.")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")

    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--load_model_from_checkpoint", default=False, action='store_true',
                        help="Whether to load model from checkpoint.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--epochs', type=int, default=1,
                        help="training epochs")
    args = parser.parse_args()
    # Setup CUDA, GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO)
    logger.warning("device: %s, n_gpu: %s",device, args.n_gpu,)
    # Set seed
    set_seed(args)

    #tokenizer = Tokenizer.from_file('./wordlevel_tokenizer/wordlevel.json') 
    tokenizer = T5Tokenizer.from_pretrained('./wordlevel_tokenizer/wordlevel.json')

    #config = T5Config.from_pretrained("t5-base")
    #config.decoder_start_token_id = config.pad_token_id
    #config = T5Config.from_pretrained(args.config_name)
    #model = T5ForConditionalGeneration(config=config)    
    #model = T5ForConditionalGeneration(config)
    config = T5Config.from_pretrained("/scratch/rinao/VulRepair/t5-base-model")
    #model = T5ForConditionalGeneration.from_pretrained("/scratch/rinao/VulRepair/t5-base")
    model = T5ForConditionalGeneration.from_pretrained("/scratch/rinao/VulRepair/t5-base-model")
    model.resize_token_embeddings(32100)
    
    logger.info("Training/evaluation parameters %s", args)
    # Training
    if args.do_train:
        train_dataset = TextDataset(tokenizer, args, file_type='train')
        eval_dataset = TextDataset(tokenizer, args, file_type='eval')
        if args.load_model_from_checkpoint:
            checkpoint_prefix = f'checkpoint-best-loss/{args.checkpoint_model_name}'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
            model.load_state_dict(torch.load(output_dir))
            model.to(args.device)
        train(args, train_dataset, model, tokenizer, eval_dataset)
    # Evaluation
    results = {}
    if args.do_eval:
        checkpoint_prefix = f'checkpoint-best-loss/{args.model_name}'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        model.load_state_dict(torch.load(output_dir))
        model.to(args.device)
        eval_dataset = TextDataset(tokenizer, args, file_type='eval')
        result=evaluate(args, model, tokenizer, eval_dataset)   
    if args.do_test:
        checkpoint_prefix = f'checkpoint-best-loss/{args.model_name}'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        model.load_state_dict(torch.load(output_dir, map_location=args.device))
        model.to(args.device)
        test_dataset = TextDataset(tokenizer, args, file_type='test')
        test(args, model, tokenizer, test_dataset, best_threshold=0.5)
    return results

if __name__ == "__main__":
    main()
