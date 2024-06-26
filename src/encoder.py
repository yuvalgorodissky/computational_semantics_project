import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AdamW, get_linear_schedule_with_warmup
from evaluating import compute_metrics
from tqdm.auto import tqdm
from  data_processing import load_data
from utils import set_seed


def train(model, tokenizer, train_dataloader, device, optimizer, scheduler):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_dataloader, desc="Training", leave=False)

    for batch in progress_bar:
        # Assuming the batch contains dictionaries with keys like 'context', 'question', 'answer', etc.
        # Prepare the inputs for the model
        inputs = tokenizer(
            batch['question'],
            batch['context'],
            truncation="only_second",  # Truncate only the context part if too long
            max_length=512,
            padding="max_length",
            return_tensors="pt"
        )
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        start_positions = torch.tensor(batch['answer_start']).to(device)
        end_positions = torch.tensor(batch['answer_end']).to(device)

        # Reset gradients
        model.zero_grad()

        # Forward pass to get output from model
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            start_positions=start_positions,
            end_positions=end_positions
        )

        # Get loss from outputs
        loss = outputs.loss
        # Backward pass to calculate gradients
        loss.backward()

        # Update model parameters
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / len(train_dataloader)
    return avg_loss

def evaluate(model, tokenizer, dataloader, device):
    model.eval()
    all_predictions = {}
    all_answers = {}
    progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)

    for batch in progress_bar:
        inputs = tokenizer(
            batch['question'],
            batch['context'],
            truncation="only_second",
            max_length=512,
            padding="max_length",
            return_tensors="pt"
        )
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            start_logits, end_logits = outputs.start_logits, outputs.end_logits

        # Convert logits to actual predictions
        for i in range(len(batch['id'])):
            start_index = torch.argmax(start_logits[i]).item()
            end_index = torch.argmax(end_logits[i]).item()
            pred_answer = tokenizer.convert_tokens_to_string(
                tokenizer.convert_ids_to_tokens(
                    input_ids[i, start_index:end_index+1],
                    skip_special_tokens=True
                )
            )

            all_predictions[batch['id'][i]] = pred_answer.strip()
            all_answers[batch['id'][i]] = batch['answer'][i].strip()
    prediction_answers = [(all_predictions[key],all_answers[key]) for key in all_answers.keys()]
    predicitons , ground_truth = zip(*prediction_answers)
    metrics = compute_metrics(predicitons,ground_truth)
    return metrics,all_predictions



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
    parser.add_argument("--output_dir", type=str, default="/sise/eliorsu-group/yuvalgor/courses/computational_semantics_project/models/bert-large-uncased", help="Output path for saved model and metrics")
    parser.add_argument("--model_name_or_path", type=str, default="google-bert/bert-large-uncased", help="Model name or path")

    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    return args


def main():
    args = parser()
    set_seed(args.seed)  # For reproducibility
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_name_or_path)
    model = model.to(device)
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
            model = AutoModelForQuestionAnswering.from_pretrained(args.output_dir)
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
