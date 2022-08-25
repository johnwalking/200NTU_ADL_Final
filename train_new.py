import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
#os.environ['CUDA_VISIBLE_DEVICES'] = "5"
import csv
import json
import torch
import argparse
import numpy as np
import torch.nn as nn
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    AutoModelForCausalLM, 
    DataCollatorForSeq2Seq,
    default_data_collator, 
    Trainer, 
    AutoModelForSeq2SeqLM, 
    T5ForConditionalGeneration,
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments, 
    TrainerCallback,
)
from datasets import load_metric, load_dataset
from transformers.trainer_callback import EarlyStoppingCallback
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", 
        default="facebook/blenderbot-400M-distill", 
        type=str, help="model to finetune")
    parser.add_argument("--output_dir", default='sim-fb2', 
        type=str, help="dir to save finetuned model")
    parser.add_argument("--train_file", default='./simulator_train.csv', type=str)
    parser.add_argument("--valid_file", default='./simulator_valid.csv', type=str)
    parser.add_argument("--ignore_pad_token_for_loss", default=True, type=bool)
    parser.add_argument("--do_train", default=True, type=bool)
    parser.add_argument("--do_eval", default=True, type=bool)
    parser.add_argument("--max_epoch", default=10, type=int, help="total number of epoch")
    parser.add_argument("--max_len", default=256, type=int, help="maximum sequence length")
    parser.add_argument("--batch_size", default=8, type=int, help="training batch size")
    parser.add_argument("--grad_step", default=4, type=int, help="gradient accumulation step")
    parser.add_argument("--eval_steps", default=1000, type=int)
    parser.add_argument("--evaluation_strategy", default="steps", type=str)
    parser.add_argument("--patience", default=3, type=int, help="early stopping patience")
    parser.add_argument("--device", default="cuda:0", type=torch.device, help="cpu, cuda, cuda:0, cuda:1")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    # Set up tokenizer first
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    # tokenizer = BlenderbotTokenizer.from_pretrained(args.model_name_or_path)

    # Prepare function for data preprocessing
    def preprocess_function(examples):
        inputs = [ex for ex in examples['inputs']]
        targets = [ex for ex in examples['target']]
        model_inputs = tokenizer(
            inputs, max_length=args.max_len, truncation=True, 
            padding='max_length', add_special_tokens=True)
        # Set up the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets, max_length=args.max_len, truncation=True,# return_tensors="pt",
                padding='max_length', add_special_tokens=True)
        model_inputs["labels"] = labels["input_ids"]
        # model_inputs["decoder_input_ids"] = labels["input_ids"]
        return model_inputs

    # Dataset and preprocessing
    print('reading dataset')
    datasets = {}
    extension = "csv"
    train_data_path = os.path.join(args.train_file)
    datasets["train"] = load_dataset(extension, data_files=[train_data_path], use_auth_token=True)
    datasets["train"] = datasets["train"].map(
        preprocess_function, batched=True, remove_columns=['inputs', 'target'],
    )["train"]
    eval_data_path = os.path.join(args.valid_file)
    datasets["valid"] = load_dataset(extension, data_files=[eval_data_path], use_auth_token=True)
    datasets["valid"] = datasets["valid"].map(
        preprocess_function, batched=True, remove_columns=['inputs', 'target'],
    )["train"]
    print(datasets)

    train_dataset, eval_dataset = datasets["train"], datasets["valid"]
    model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
    # model.resize_token_embeddings(len(tokenizer))
    # model = BlenderbotForConditionalGeneration.from_pretrained(args.model_name_or_path).to(args.device)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100)
    metric_bert = load_metric("bertscore")
    metric_bleu = load_metric("sacrebleu")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]
        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
            
        # Calculate BERT Score
        result = metric_bert.compute(predictions=decoded_preds, 
                                     references=decoded_labels, lang="en")
        from statistics import mean
        result["precision"] = mean(result["precision"])
        result["recall"] = mean(result["recall"])
        result["f1"] = mean(result["f1"])

        # Calculate BLEU Score
        result_bleu = metric_bleu.compute(predictions=decoded_preds, references=decoded_labels)
        result["sacrebleu"] = result_bleu["score"]
        
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items() if type(v) == float}
        return result

    # Train model
    training_args = Seq2SeqTrainingArguments(
        output_dir = args.output_dir,
        overwrite_output_dir = False,
        do_train = args.do_train,
        do_eval = args.do_eval,
        logging_strategy = "epoch",
        save_strategy = "epoch",
        num_train_epochs = args.max_epoch,
        gradient_accumulation_steps = args.grad_step,
        per_device_train_batch_size = args.batch_size,
        eval_accumulation_steps = args.grad_step,
        per_device_eval_batch_size = args.batch_size,
        label_smoothing_factor = 0.1,
        predict_with_generate=True,
        eval_steps = args.eval_steps,
        evaluation_strategy = args.evaluation_strategy,
    )
    
    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model = model,
        args = training_args,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        tokenizer = tokenizer,
        data_collator = data_collator,
        compute_metrics = compute_metrics,
    )

    # Detecting last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train \
        and not training_args.overwrite_output_dir:
        from transformers.trainer_utils import get_last_checkpoint
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None and \
            training_args.resume_from_checkpoint is None:
            print("Checkpoint detected, resuming training at {last_checkpoint}.")

    # Training
    if args.do_train:
        checkpoint = None
        # if training_args.resume_from_checkpoint is not None:
        #    checkpoint = training_args.resume_from_checkpoint
        # elif 
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        # train_result = trainer.train()
        trainer.save_model()
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if args.do_eval:
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
