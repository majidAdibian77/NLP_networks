
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoConfig
from transformers import TrainingArguments, Trainer
import argparse
import os

from prepare_data import read_data, read_and_add_squad_data, tokenize_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", required=True, type=str, help="type of model")
    args = parser.parse_args()

    if args.model_type == "pretrained_parsbert":
        tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/bert-fa-base-uncased")
        model = AutoModelForQuestionAnswering.from_pretrained("HooshvareLab/bert-fa-base-uncased")
    elif args.model_type == "untrained_bert":
        config = AutoConfig.from_pretrained("bert-base-uncased")
        model = AutoModelForQuestionAnswering.from_config(config)
        tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/bert-fa-base-uncased")

    elif args.model_type == "pretrained_xlm_roberta":
        tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
        model = AutoModelForQuestionAnswering.from_pretrained("xlm-roberta-base")
    else:
        print("--model_type must be one of <pretrained_parsbert>, <untrained_bert>, <pretrained_xlm_roberta>")
        return -1

    train_data = read_data(os.path.join('data/pquad','train.json'))
    val_data = read_data(os.path.join('data/pquad','val.json'))

    if args.model_type == "pretrained_xlm_roberta":
        train_data, val_data = read_and_add_squad_data(os.path.join('data/squad','train.json'), train_data, val_data)

    tokenized_train_data = tokenize_data(tokenizer, train_data, 512, 256)
    print("train data is tokenized")
    tokenized_val_data = tokenize_data(tokenizer, val_data, 512, 256)
    print("validation data is tokenized")

    args = TrainingArguments(
    f"models/{args.model_type}",
    evaluation_strategy = "epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    save_steps=100,
    eval_steps=100,
    num_train_epochs=200,
    weight_decay=0.0001) 

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_train_data,
        eval_dataset=tokenized_val_data,
        tokenizer=tokenizer)

    # start training
    trainer.train()

if __name__ == "__main__":
    main()
