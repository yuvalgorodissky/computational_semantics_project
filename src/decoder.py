import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
    AdamW,
    DataCollatorForLanguageModeling,
)

from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from data_processing import load_data
from evaluating import compute_metrics
import argparse
import json
from tqdm import tqdm
from utils import set_seed, clean_output
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM


def formatting_prompts_func(data_point, for_test=False):
    if not for_test:
        answer = data_point["answer"] if data_point["answer"] else "unanswerable"

        data_point["text"] = f"""Give an answer to the question given the context:  
    ### Context: {data_point["context"]}
    ### Question: {data_point["question"]}
    ### Answer: {answer}"""
    else:
        data_point["text"] = f"""Give an answer to the question given the context:  
          ### Context: {data_point["context"]}
          ### Question: {data_point["question"]}
          ### Answer:"""

    return data_point


class SquadDataset(Dataset):
    def __init__(self, entries, tokenizer, transform=None, for_test=False):
        super(SquadDataset, self).__init__()
        self.tokenizer = tokenizer
        self.transform = transform
        entries_after_transform = []
        for entry in entries:
            entries_after_transform.append(self.transform(entry, for_test))
        self.entries = entries_after_transform

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]["text"]

        tokenizer_text = self.tokenizer(entry, return_tensors="pt", padding="max_length", truncation=True,
                                        max_length=1024)

        for key in tokenizer_text:
            tokenizer_text[key] = tokenizer_text[key].squeeze(0)  # Remove batch dimension if added automatically
        return tokenizer_text


def evaluate(model, tokenizer, dataset,device):
    model.eval()
    predictions_dict = {}
    predictions = []
    with torch.no_grad():
        for i ,data in enumerate(tqdm(dataset, desc="Evaluating")):
            input_ids = data["input_ids"].to(device).unsqueeze(0)
            attention_mask = data["attention_mask"].to(device).unsqueeze(0)

            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                     early_stopping=True, max_new_tokens=20)

            pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            # Decode the input tokens
            #TODO: check if this is the correct way to decode the input extarct the answer from the genarated text 
            # Subtract the input part from the full output to get only the generated part
            generated_part_start = pred_text.find(decoded_input) + len(decoded_input)
            generated_part = pred_text[generated_part_start:].strip()
            clean_prediction = clean_output(generated_part)
            predictions.append(clean_prediction)
            predictions_dict[data['qid']] = generated_part
    with open(path, "w") as f:
        json.dump(predictions_dict, f)
    metrics = compute_metrics(predictions, references)
    print(f"Evaluated metrics: {metrics}")
    return metrics


def setup_training_arguments(args):
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size_train,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        save_steps=0,
        logging_steps=5,
        learning_rate=args.lr,
        fp16=False,
        bf16=False,
        lr_scheduler_type="cosine",
        group_by_length=True,
        max_grad_norm=0.3,
        max_steps=-1,
        weight_decay=0.001,
        warmup_ratio=0.03,

    )
    return training_args


def get_lora_params(model):
    lora_params = [p for n, p in model.named_parameters() if 'lora' in n]
    return lora_params


def parser():
    parser = argparse.ArgumentParser(description="Train and evaluate a T5 model on SQuAD 2.0")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--batch_size_train", type=int, default=8, help="Batch size for training")
    parser.add_argument("--batch_size_dev", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--path_dev_set", type=str, default="", required=False, help="Path to the dev set")
    parser.add_argument("--path_train_set", type=str, default="", required=False, help="Path to the train set")
    parser.add_argument("--do_eval", action='store_true', help="Perform evaluation")
    parser.add_argument("--do_train", action='store_true', help="Perform training")
    parser.add_argument("--output_dir", type=str,
                        default="/sise/eliorsu-group/yuvalgor/courses/computational_semantics_project/models/bert-large-uncased",
                        help="Output path for saved model and metrics")
    parser.add_argument("--model_name_or_path", type=str, default="google-bert/bert-large-uncased",
                        help="Model name or path")
    parser.add_argument('--lora_alpha', type=int, default=64, help='Alpha parameter for LoRA scaling')
    parser.add_argument('--lora_r', type=int, default=64, help='LoRA attention dimension')
    parser.add_argument('--lora_dropout', type=float, default=0.1, help='Dropout probability for LoRA layers')
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    return args


class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, mlm=False, mlm_probability=0.15):
        super().__init__(tokenizer=tokenizer, mlm=mlm, mlm_probability=mlm_probability)

    def __call__(self, examples):
        # Collate examples using the parent class method
        batch = super().__call__(examples)

        # Tokenize "Answer:" to find where the answer starts
        answer_start_tokens = self.tokenizer(" Answer:", add_special_tokens=False)['input_ids']

        # Adjust labels to set non-answer tokens to -100
        for i, input_ids in enumerate(batch['input_ids']):
            input_ids_list = input_ids.tolist()  # Convert tensor to list

            # Try to find the sequence of answer_start_tokens in input_ids_list
            start_answer = None
            for index in range(len(input_ids_list) - len(answer_start_tokens) + 1):
                if input_ids_list[index:index + len(answer_start_tokens)] == answer_start_tokens:
                    start_answer = index + len(answer_start_tokens)
                    break

            if start_answer:
                # Assume answer extends until the next special token (e.g., pad, eos) or the end of the list
                end_answer = next((idx for idx, token in enumerate(input_ids_list[start_answer:], start=start_answer)
                                   if token in [self.tokenizer.pad_token_id, self.tokenizer.eos_token_id]),
                                  len(input_ids_list))

                # Set non-answer labels to -100
                labels = batch['labels'][i].tolist()  # Convert tensor to list
                labels[:start_answer] = [-100] * start_answer  # Set tokens before answer to -100
                labels[end_answer:] = [-100] * (len(labels) - end_answer)  # Set tokens after answer to -100
                batch['labels'][i] = torch.tensor(labels)  # Convert back to tensor
            else:
                # If no answer start is found, set all labels to -100
                batch['labels'][i] = torch.full_like(batch['labels'][i], -100)

        return batch


def prepare_bnb_config():
    use_4bit = True
    bnb_4bit_compute_dtype = "float16"
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )
    return bnb_config


def main():
    args = parser()
    set_seed(args.seed)  # Ensure reproducibility
    logging.set_verbosity_info()  # Set logging to show info messages
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Configuring model and tokenizer...")
    bnb_config = prepare_bnb_config()

    # Load model with the configuration that handles device placement
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto",  # Automatically handle device placement
        torch_dtype=torch.bfloat16,
        token="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",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Further configuration (e.g., LoRA) if needed
    peft_config = LoraConfig(
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "up_proj", "o_proj", "k_proj", "down_proj", "gate_proj", "v_proj"],
        r=args.lora_r,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)

    model.to(device)
    if args.do_train:
        model.print_trainable_parameters()
        print(f"Loading training data from dir -- {args.path_train_set} -- ...")
        train_data = load_data(args.path_train_set)[:10]
        train_dataset = SquadDataset(train_data, tokenizer, transform=formatting_prompts_func)
        # train_dataset = train_dataset.map(formatting_prompts_func)
        training_args = setup_training_arguments(args)

        collator = DataCollatorForCompletionOnlyLM(tokenizer=tokenizer)
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            dataset_text_field="text",
            peft_config=peft_config,
            max_seq_length=1024,
            args=training_args,
            packing=False,
            data_collator=collator,
            # optimizers=(AdamW(get_lora_params(model), lr=args.learning_rate), None)  # Only LoRA params
        )
        trainer.train()
        print("Training complete. Saving model...")

        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"Model saved to {args.output_dir}")

    if args.do_eval:
        print(f"Loading evaluation data from {args.path_dev_set}...")
        dev_data = load_data(args.path_dev_set)
        references = [entry["answer"] for entry in dev_data]
        dev_dataset = SquadDataset(dev_data, tokenizer, transform=formatting_prompts_func, for_test=True)
        metrics, all_predictions = evaluate(model, tokenizer, dev_dataset,device)
        print("Evaluation complete.")


        # with open(f"{args.output_dir}/metrics.json", 'w') as f:
        #     json.dump(metrics, f)
        # with open(f"{args.output_dir}/all_predictions.json", 'w') as f:
        #     json.dump(all_predictions, f, indent=4)
        # print("Metrics and predictions saved.")

        print("Evaluation complete.")


if __name__ == "__main__":
    main()
