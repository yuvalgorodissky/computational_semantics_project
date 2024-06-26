import argparse
import json
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW, get_linear_schedule_with_warmup, AutoTokenizer, \
    AutoModelForSeq2SeqLM
from data_processing import load_data
from evaluating import compute_metrics
from tqdm.auto import tqdm
from utils import set_seed


def parser():
    parser = argparse.ArgumentParser(description="Train and evaluate a T5 model on SQuAD 2.0")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--batch_size_train", type=int, default=8, help="Batch size for training")
    parser.add_argument("--batch_size_dev", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--path_dev_set", type=str,default="", required=False, help="Path to the dev set")
    parser.add_argument("--path_train_set", type=str,default="", required=False, help="Path to the train set")
    parser.add_argument("--do_eval", action='store_true', help="Perform evaluation")
    parser.add_argument("--do_train", action='store_true', help="Perform training")
    parser.add_argument("--output_dir", type=str, default="google/flan-t5-base", help="Output path for saved model and metrics")
    parser.add_argument("--model_name_or_path", type=str, default="google-bert/bert-large-uncased", help="Model name or path")

    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    return args

def evaluate(model, tokenizer, dataloader, device,max_lenght=512):
    model.eval()
    predictions = []
    references = []
    predictions_dict = {}
    for batch in tqdm(dataloader, desc="Generating predictions"):
        question_texts = batch['question']
        context_texts = batch['context']
        inputs = tokenizer(context_texts, question_texts, padding=True, truncation=True, max_length=max_lenght,
                           return_tensors="pt")
        qids = batch['id']
        inputs = {key: val.to(device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(**inputs)

        batch_predictions = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in
                             outputs]
        predictions.extend(batch_predictions)
        references.extend(batch['answer'])
        for qid, prediction in zip(qids, batch_predictions):
            predictions_dict[qid] = prediction.replace('<extra_id_0>', '')

            # Ensure removal in dict as well

    metrics = compute_metrics(predictions, references)
    return metrics, predictions_dict


def train(model, tokenizer, train_dataloader, device, optimizer, scheduler,max_length=512):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_dataloader, desc="Training", leave=False)

    for batch in progress_bar:
        optimizer.zero_grad()

        question_texts = batch['question']
        context_texts = batch['context']
        target_texts = batch['answer']

        inputs = tokenizer(context_texts, question_texts, padding=True, truncation=True, max_length=max_length,
                           return_tensors="pt")
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(target_texts, padding=True, truncation=True, max_length=max_length,
                               return_tensors="pt").input_ids

        inputs = {key: val.to(device) for key, val in inputs.items()}
        labels = labels.to(device)
        labels[labels == tokenizer.pad_token_id] = -100

        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()

        total_loss += loss.item()
        optimizer.step()
        scheduler.step()

        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / len(train_dataloader)
    return avg_loss  # Return average loss for printing

def main():
    args = parser()
    set_seed(args.seed)  # For reproducibility
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
    model.to(device)
    if args.do_train:
        print(f"Loading training data from dir -- {args.path_train_set} -- ...")

        train_data = load_data(args.path_train_set)
        train_dataloader = DataLoader(train_data, batch_size=args.batch_size_train, shuffle=True)

        optimizer = AdamW(model.parameters(), lr=args.lr)
        total_steps = len(train_dataloader) * args.epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
        print("Starting training...")
        for epoch in range(args.epochs):
            print(f"Epoch {epoch + 1}/{args.epochs}")
            avg_loss = train(model, tokenizer, train_dataloader, device, optimizer, scheduler)
            print(f"Epoch {epoch + 1}/{args.epochs} - Average Training Loss: {avg_loss:.4f}")

        print("Training complete. Saving model...")

        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"Model saved to {args.output_dir}")

    if args.do_eval:
        if args.do_train:
            print("Loading model from output directory...")
            tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
            model = AutoModelForSeq2SeqLM.from_pretrained(args.output_dir)
            model = model.to(device)
        print(f"Loading dev data from dir -- {args.path_dev_set} -- ...")
        dev_data = load_data(args.path_dev_set)
        dev_dataloader = DataLoader(dev_data, batch_size=args.batch_size_dev, shuffle=False)
        print("Starting evaluation...")

        metrics, all_predictions = evaluate(model, tokenizer, dev_dataloader, device)
        print("Final Results:")
        print(metrics)
        # Save metrics and predictions to JSON files
        with open(f"{args.output_dir}/metrics.json", 'w') as f:
            json.dump(metrics, f)
        with open(f"{args.output_dir}/all_predictions.json", 'w') as f:
            json.dump(all_predictions, f, indent=4)
        print("Metrics and predictions saved.")

        print("Evaluation complete.")

if __name__ == "__main__":
    main()
