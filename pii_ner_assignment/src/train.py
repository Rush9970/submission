import os
import json
import torch
import tempfile
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from dataset import PIIDataset, collate_batch
from labels import LABELS, ID2LABEL, label_is_pii
from model import create_model
from eval_span_f1 import load_gold, compute_prf


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="distilbert-base-uncased")
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--dev", default="data/dev.jsonl")
    ap.add_argument("--out_dir", default="out")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def predict(model, tokenizer, dataset, device, max_length=256):
    """Generate predictions for the given dataset."""
    model.eval()
    all_preds = {}
    
    for item in dataset:
        uid = item["id"]
        text = item["text"]
        
        # Tokenize input
        encodings = tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        # Move to device
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)
        offsets = encodings["offset_mapping"][0].tolist()
        
        # Get predictions
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            pred_ids = outputs.logits.argmax(dim=-1)[0].cpu().numpy()
        
        # Convert to spans
        spans = []
        current_label = None
        current_start = None
        current_end = None
        
        for (start, end), pred_id in zip(offsets, pred_ids):
            if start == 0 and end == 0:  # Skip special tokens
                continue
                
            label = ID2LABEL.get(pred_id, "O")
            
            if label == "O":
                if current_label is not None:
                    spans.append({
                        "start": current_start,
                        "end": current_end,
                        "label": current_label,
                        "pii": bool(label_is_pii(current_label))
                    })
                    current_label = None
                continue
                
            prefix, ent_type = label.split("-", 1)
            
            if prefix == "B" or (prefix == "I" and (current_label is None or current_label != ent_type)):
                if current_label is not None:
                    spans.append({
                        "start": current_start,
                        "end": current_end,
                        "label": current_label,
                        "pii": bool(label_is_pii(current_label))
                    })
                current_label = ent_type
                current_start = start
                current_end = end
            elif prefix == "I" and current_label == ent_type:
                current_end = end
        
        if current_label is not None:
            spans.append({
                "start": current_start,
                "end": current_end,
                "label": current_label,
                "pii": bool(label_is_pii(current_label))
            })
        
        all_preds[uid] = spans
    
    return all_preds

def evaluate_pii_precision(model, tokenizer, eval_file, device, max_length=256):
    """Evaluate the model and return PII precision."""
    # Load gold data
    gold = load_gold(eval_file)
    
    # Create dataset for prediction
    eval_ds = []
    with open(eval_file, 'r', encoding='utf-8') as f:
        for line in f:
            eval_ds.append(json.loads(line))
    
    # Get predictions
    preds = predict(model, tokenizer, eval_ds, device, max_length)
    
    # Calculate PII precision
    pii_tp = pii_fp = 0
    
    for uid in gold:
        # Get gold PII spans
        gold_spans = set()
        for start, end, label in gold[uid]:
            if label_is_pii(label):
                gold_spans.add((start, end, "PII"))
        
        # Get predicted PII spans
        pred_spans = set()
        for span in preds.get(uid, []):
            if span["pii"]:
                pred_spans.add((span["start"], span["end"], "PII"))
        
        # Update TP and FP
        for span in pred_spans:
            if span in gold_spans:
                pii_tp += 1
            else:
                pii_fp += 1
    
    # Calculate precision
    pii_prec = pii_tp / (pii_tp + pii_fp) if (pii_tp + pii_fp) > 0 else 0.0
    return pii_prec

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Load tokenizer and datasets
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_ds = PIIDataset(args.train, tokenizer, LABELS, max_length=args.max_length, is_train=True)
    
    # Create data loaders
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_batch(b, pad_token_id=tokenizer.pad_token_id),
    )

    # Initialize model and optimizer
    model = create_model(args.model_name)
    model.to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # Learning rate scheduler
    total_steps = len(train_dl) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.1 * total_steps), 
        num_training_steps=total_steps
    )

    # Training loop with early stopping
    best_pii_prec = 0.0
    patience = 2
    no_improve = 0

    for epoch in range(args.epochs):
        # Training
        model.train()
        running_loss = 0.0
        
        for batch in tqdm(train_dl, desc=f"Epoch {epoch+1}/{args.epochs}"):
            input_ids = torch.tensor(batch["input_ids"], device=args.device)
            attention_mask = torch.tensor(batch["attention_mask"], device=args.device)
            labels = torch.tensor(batch["labels"], device=args.device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

        # Calculate average training loss
        avg_loss = running_loss / max(1, len(train_dl))
        print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")
        
        # Evaluate on dev set
        pii_prec = evaluate_pii_precision(
            model, tokenizer, args.dev, args.device, args.max_length
        )
        print(f"Dev PII precision: {pii_prec:.4f}")
        
        # Check for improvement
        if pii_prec > best_pii_prec + 1e-6:  # Small threshold to avoid floating point issues
            best_pii_prec = pii_prec
            no_improve = 0
            
            # Save best model
            model.save_pretrained(args.out_dir)
            tokenizer.save_pretrained(args.out_dir)
            print(f"New best model saved with PII precision: {best_pii_prec:.4f}")
        else:
            no_improve += 1
            print(f"No improvement in PII precision for {no_improve}/{patience} epochs")
            
            # Early stopping
            if no_improve >= patience:
                print(f"Early stopping after {epoch+1} epochs")
                break
    
    print(f"Training complete. Best PII precision: {best_pii_prec:.4f}")
    print(f"Model saved to {args.out_dir}")


if __name__ == "__main__":
    main()
