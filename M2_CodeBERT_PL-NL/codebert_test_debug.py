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
import glob
import logging
import os
import pickle
import random
import re
import numpy as np
import torch
import torch.nn as nn
from codebert_model import Seq2Seq
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup, 
                          RobertaConfig, RobertaModel, RobertaTokenizer)
from tqdm import tqdm
import multiprocessing
import pandas as pd
import datasets
from sklearn.model_selection import train_test_split

cpu_cont = 16
logger = logging.getLogger(__name__)

#### CHANGES ####
def get_next_run_filename(base_path, prefix="m2_test_debug_run_", extension=".txt"):
    existing_files = os.listdir(base_path)
    run_nums = []
    pattern = re.compile(f"{re.escape(prefix)}(\\d+){re.escape(extension)}")
    for file in existing_files:
        match = pattern.match(file)
        if match:
            run_nums.append(int(match.group(1)))
    next_run = max(run_nums) + 1 if run_nums else 1
    return os.path.join(base_path, f"{prefix}{next_run}{extension}")
#### CHANGES ####

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
    def __init__(self, tokenizer, args, train_data=None, val_data=None, file_type="train"):
        if file_type == "train":
            sources = train_data["source"].tolist()
            labels = train_data["target"].tolist()
        elif file_type == "eval":
            sources = val_data["source"].tolist()
            labels = val_data["target"].tolist()
        # elif file_type == "test":
        #     data = datasets.load_dataset("MickyMike/cvefixes_bigvul", split="test")
        #     sources = data["source"]
        #     labels = data["target"]
        elif file_type == "test":
            test_df = pd.read_csv("../data/cvefixes_bigvul/test.csv")
            sources = test_df["source"].tolist()
            labels = test_df["target"].tolist()
        self.examples = []
        for i in tqdm(range(len(sources))):
            #self.examples.append(convert_examples_to_features(sources[i], labels[i], tokenizer, args))
            feature = convert_examples_to_features(sources[i], labels[i], tokenizer, args)
            # STORE INDEX
            #feature.index = i  # Attach index
            self.examples.append(feature)
    
        if file_type == "train":
            for example in self.examples[:3]:
                    logger.info("*** Example ***")
                    logger.info("label: {}".format(example.label))
                    logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))
                    logger.info("decoder_input_ids: {}".format(' '.join(map(str, example.decoder_input_ids))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):       
        # return self.examples[i].input_ids, self.examples[i].input_ids.ne(1), self.examples[i].label, self.examples[i].decoder_input_ids, self.examples[i].decoder_input_ids.ne(1), self.examples[i].index
        return self.examples[i].input_ids, self.examples[i].input_ids.ne(1), self.examples[i].label, self.examples[i].decoder_input_ids, self.examples[i].decoder_input_ids.ne(1)
        # added self.examples[i].index

def convert_examples_to_features(source, label, tokenizer, args):
    # encode
    source_ids = tokenizer.encode(source, truncation=True, max_length=args.encoder_block_size, padding='max_length', return_tensors='pt')
    decoder_input_ids = tokenizer.encode(label, truncation=True, max_length=args.decoder_block_size, padding='max_length', return_tensors='pt')
    label = tokenizer.encode(label, truncation=True, max_length=args.decoder_block_size, padding='max_length', return_tensors='pt')
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
    # evaluate the model per epoch
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

    model.zero_grad()

    for idx in range(args.epochs): 
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        tr_num = 0
        train_loss = 0
        for step, batch in enumerate(bar):
            (input_ids, attention_mask, labels, decoder_input_ids, target_mask) = [x.squeeze(1).to(args.device) for x in batch]
            model.train()
            # the forward function automatically creates the correct decoder_input_ids
            loss, _, _ = model(source_ids=input_ids, source_mask=attention_mask, target_ids=decoder_input_ids, target_mask=target_mask)
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
                avg_loss = round(np.exp((tr_loss - logging_loss) /(global_step- tr_nb)), 4)
                if global_step % args.save_steps == 0:
                    # placeholder of evaluation
                    result = evaluate(args, model, tokenizer, eval_dataset, eval_when_training=True)    
                    # Save model checkpoint
                    if result < best_loss:
                        best_loss = result
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
    tokens = tokens.replace("<pad>", "")
    tokens = tokens.replace("<s>", "")
    tokens = tokens.replace("</s>", "")
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
    bar = tqdm(eval_dataloader, total=len(eval_dataloader))
    
    eval_loss, tokens_num = 0, 0
    for batch in bar:
        (input_ids, attention_mask, labels, decoder_input_ids, target_mask) = [x.squeeze(1).to(args.device) for x in batch]
        with torch.no_grad():
            _, loss, num = model(source_ids=input_ids,source_mask=attention_mask,
                                target_ids=decoder_input_ids,target_mask=target_mask)     
        eval_loss += loss.sum().item()
        tokens_num += num.sum().item()
    eval_loss = eval_loss / tokens_num
    # show loss of dev dataset    
    model.train()
    logger.info("***** Eval results *****")
    logger.info(f"Evaluation Loss: {str(eval_loss)}")    
    return eval_loss

def test(args, model, tokenizer, test_dataset, best_threshold=0.5):
    # build dataloader
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size, num_workers=0)
    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    # Test!
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    model.eval()
    accuracy = []
    raw_predictions = []
    correct_prediction = ""
    bar = tqdm(test_dataloader, total=len(test_dataloader))
    for batch in bar:
        correct_pred = False
        (input_ids, attention_mask, labels, decoder_input_ids, target_mask) = [x.squeeze(1).to(args.device) for x in batch]
        with torch.no_grad():
            beam_outputs = model(source_ids=input_ids, source_mask=attention_mask)
            beam_outputs = beam_outputs.detach().cpu().tolist()[0]
        decoder_input_ids = decoder_input_ids.detach().cpu().tolist()
        for single_output in beam_outputs:
            # pred
            prediction = tokenizer.decode(single_output, skip_special_tokens=False)
            prediction = clean_tokens(prediction)
            # truth
            ground_truth = tokenizer.decode(decoder_input_ids[0], skip_special_tokens=False)
            ground_truth = clean_tokens(ground_truth)
            if prediction == ground_truth:
                correct_prediction = prediction
                correct_pred = True
                break
        if correct_pred:
            raw_predictions.append(correct_prediction)
            accuracy.append(1)
        else:
            # if not correct, use the first output in the beam as the raw prediction
            raw_pred = tokenizer.decode(beam_outputs[0], skip_special_tokens=False)
            raw_pred = clean_tokens(raw_pred)
            raw_predictions.append(raw_pred)
            accuracy.append(0)
    # calculate accuracy
    test_result = round(sum(accuracy) / len(accuracy), 4)
    logger.info("***** Test results *****")
    logger.info(f"Test Accuracy: {str(test_result)}")

def test_debug(args, model, tokenizer, test_dataset, sample_size=3):
    # test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset),batch_size=1, num_workers=0)

    ## CHANGED TO RANDOM SAMPLING ##
    test_dataloader = DataLoader(test_dataset, sampler=RandomSampler(test_dataset), batch_size=1, num_workers=0)
    ###
    
    model.eval()
    logger.info("***** Debugging Inference on Small Sample *****")

    count = 0
    for batch in test_dataloader:
        if count >= sample_size:
            break

        # added index
        # (input_ids, attention_mask, labels, decoder_input_ids, target_mask, index) = [x.squeeze(1).to(args.device) for x in batch]
        (input_ids, attention_mask, labels, decoder_input_ids, target_mask) = [x.squeeze(1).to(args.device) for x in batch]
        with torch.no_grad():
            beam_outputs = model(source_ids=input_ids, source_mask=attention_mask)
            beam_outputs = beam_outputs.detach().cpu().tolist()[0]

        decoder_input_ids_cpu = decoder_input_ids.detach().cpu().tolist()[0]

        for single_output in beam_outputs:
            prediction = tokenizer.decode(single_output, skip_special_tokens=False)
            prediction_clean = clean_tokens(prediction)

            ground_truth = tokenizer.decode(decoder_input_ids_cpu, skip_special_tokens=False)
            ground_truth_clean = clean_tokens(ground_truth)

            logger.info("="*60)
            #logger.info(f"Sample #{count + 1}")
            
            # added row index
            # logger.info(f"Sample #{count + 1} (Dataset Row #{index.item()})")
            
            logger.info(f"Correct Prediction? {'✅ YES' if prediction_clean == ground_truth_clean else '❌ NO'}")
            logger.info(f"\n🔹 Input IDs:\n{input_ids[0].tolist()}")
            logger.info(f"\n🔹 Input Tokens:\n{tokenizer.decode(input_ids[0], skip_special_tokens=False)}")
            logger.info(f"\n🔹 Decoder Input IDs:\n{decoder_input_ids_cpu}")
            logger.info(f"\n🔹 Decoder Input Tokens:\n{tokenizer.decode(decoder_input_ids[0], skip_special_tokens=False)}")
            logger.info(f"\n🔹 Prediction:\n{prediction_clean}")
            logger.info(f"\n🔹 Ground Truth:\n{ground_truth_clean}")
            logger.info("="*60)

            break  # Only log the first prediction in the beam

        count += 1

def main():
    parser = argparse.ArgumentParser()
    # Params
    parser.add_argument("--output_dir", default=None, type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    ## Other parameters
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
    parser.add_argument("--beam_size", default=50, type=int,
                        help="Beam size to use when decoding.")                          
    parser.add_argument("--model_name", default="model.bin", type=str,
                        help="Saved model name.")
    parser.add_argument("--checkpoint_model_name", default="non_domain_model.bin", type=str,
                        help="Checkpoint model name.")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--use_non_pretrained_model", action='store_true', default=False,
                        help="Whether to use non-pretrained model.")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--code_length", default=256, type=int,
                        help="Optional Code input sequence length after tokenization.") 

    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--load_model_from_checkpoint", action='store_true', default=False,
                        help="Whether to load model from checkpoint.")

    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_local_explanation", default=False, action='store_true',
                        help="Whether to do local explanation. ") 
    parser.add_argument("--reasoning_method", default=None, type=str,
                        help="Should be one of 'attention', 'shap', 'lime', 'lig'")

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
    
    #### CHANGES ####
    output_txt_path = get_next_run_filename(".", prefix="m2_test_debug_run_", extension=".txt")
    file_handler = logging.FileHandler(output_txt_path)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logging.getLogger().addHandler(file_handler)
    #### CHANGES ####
    
    # Set seed
    set_seed(args)

    config = RobertaConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    tokenizer.add_tokens(["<S2SV_StartBug>", "<S2SV_EndBug>", "<S2SV_blank>", "<S2SV_ModStart>", "<S2SV_ModEnd>"])
    # build model
    encoder = RobertaModel.from_pretrained(args.model_name_or_path,config=config)   
    encoder.resize_token_embeddings(len(tokenizer))
    decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
    decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
    model=Seq2Seq(encoder=encoder,decoder=decoder,config=config,
                  beam_size=args.beam_size,max_length=args.decoder_block_size,
                  sos_id=tokenizer.cls_token_id,eos_id=tokenizer.sep_token_id)
        
    logger.info("Training/evaluation parameters %s", args)
    # Training
    if args.do_train:
        train_data_whole = datasets.load_dataset("csv", data_files="../data/cvefixes_bigvul/train.csv", split="train")
        df = pd.DataFrame({"source": train_data_whole["source"], "target": train_data_whole["target"]})
        train_data, val_data = train_test_split(df, test_size=0.1238)
        train_dataset = TextDataset(tokenizer, args, train_data, val_data, file_type='train')
        eval_dataset = TextDataset(tokenizer, args, train_data, val_data, file_type='eval')
        train(args, train_dataset, model, tokenizer, eval_dataset)
    # Evaluation
    results = {}  
    if args.do_test:
        checkpoint_prefix = f'checkpoint-best-loss/{args.model_name}'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        model.load_state_dict(torch.load(output_dir, map_location=args.device))
        model.to(args.device)
        test_dataset = TextDataset(tokenizer, args, file_type='test')
        # test(args, model, tokenizer, test_dataset, best_threshold=0.5)   
        test_debug(args, model, tokenizer, test_dataset, sample_size=3)
    return results

if __name__ == "__main__":
    main()