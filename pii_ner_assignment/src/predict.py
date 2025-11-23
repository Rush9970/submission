import json
import argparse
import re
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForTokenClassification
from labels import ID2LABEL, label_is_pii
import os

# Configuration
SPAN_CONF_THRESH = 0.60  # Tune on dev for highest PII precision

# PII Validation Helpers
MONTHS = [
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december"
]

def is_valid_email(s):
    s = s.lower()
    return (" at " in s and " dot " in s) or ("@" in s and "." in s)

def is_valid_phone(s):
    digits = "".join(ch for ch in s if ch.isdigit())
    return 8 <= len(digits) <= 12

def is_valid_credit_card(s):
    digits = "".join(ch for ch in s if ch.isdigit())
    return 13 <= len(digits) <= 16

def is_valid_person_name(s):
    return len(s.split()) >= 1 and len(s.strip()) >= 3

def is_valid_date(s):
    s = s.lower()
    return any(m in s for m in MONTHS)

def span_confidence_ok(span_token_indices, pred_conf, thresh=SPAN_CONF_THRESH):
    """Check if all tokens in span meet confidence threshold."""
    return all(pred_conf[i] >= thresh for i in span_token_indices)


def bio_to_spans(text, offsets, pred_ids, pred_conf=None):
    """Convert BIO tags to spans with optional confidence filtering."""
    spans = []
    current_label = None
    current_start = None
    current_end = None
    current_tokens = []  # Track token indices for confidence checking

    for idx, ((start, end), lid) in enumerate(zip(offsets, pred_ids)):
        if start == 0 and end == 0:  # Skip special tokens
            continue
            
        label = ID2LABEL.get(int(lid), "O")
        
        if label == "O":
            if current_label is not None:
                # Check confidence if confidence scores are provided
                if pred_conf is None or span_confidence_ok(current_tokens, pred_conf):
                    spans.append((current_start, current_end, current_label))
                current_label = None
                current_tokens = []
            continue

        prefix, ent_type = label.split("-", 1)
        
        if prefix == "B" or (current_label is not None and current_label != ent_type):
            if current_label is not None:
                if pred_conf is None or span_confidence_ok(current_tokens, pred_conf):
                    spans.append((current_start, current_end, current_label))
                current_tokens = []
            current_label = ent_type
            current_start = start
            current_end = end
            current_tokens.append(idx)
        elif prefix in ["I", "B"] and current_label == ent_type:
            current_end = end
            current_tokens.append(idx)

    # Add the last span if exists
    if current_label is not None:
        if pred_conf is None or span_confidence_ok(current_tokens, pred_conf):
            spans.append((current_start, current_end, current_label))

    return spans


def validate_pii_span(span_text, label):
    """Apply PII-specific validation rules."""
    if not span_text or not label:
        return False
        
    text_lower = span_text.lower()
    
    if label == "EMAIL":
        return is_valid_email(text_lower)
    elif label == "PHONE":
        return is_valid_phone(text_lower)
    elif label == "CREDIT_CARD":
        return is_valid_credit_card(text_lower)
    elif label == "PERSON_NAME":
        return is_valid_person_name(span_text)  # Use original case for names
    elif label == "DATE":
        return is_valid_date(text_lower)
    # For non-PII labels or labels without specific validation rules
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="out")
    ap.add_argument("--model_name", default=None)
    ap.add_argument("--input", default="data/dev.jsonl")
    ap.add_argument("--output", default="out/dev_pred.json")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--conf_threshold", type=float, default=SPAN_CONF_THRESH,
                   help="Minimum confidence threshold for span acceptance")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir if args.model_name is None else args.model_name)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    model.to(args.device)
    model.eval()

    results = {}
    total_spans = 0
    filtered_spans = 0

    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text = obj["text"]
            uid = obj["id"]

            # Tokenize input
            enc = tokenizer(
                text,
                return_offsets_mapping=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            offsets = enc["offset_mapping"][0].tolist()
            input_ids = enc["input_ids"].to(args.device)
            attention_mask = enc["attention_mask"].to(args.device)

            # Get model predictions
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                # Get probabilities and predictions
                probs = F.softmax(logits, dim=-1)[0]  # [seq_len, num_labels]
                pred_conf = probs.max(dim=-1).values.cpu().numpy()  # Confidence scores
                pred_ids = probs.argmax(dim=-1).cpu().numpy()  # Predicted label IDs

            # Get spans with confidence filtering
            spans = bio_to_spans(text, offsets, pred_ids, pred_conf)
            
            # Process and validate spans
            ents = []
            for s, e, lab in spans:
                total_spans += 1
                span_text = text[s:e]
                
                # Skip if PII validation fails
                if label_is_pii(lab) and not validate_pii_span(span_text, lab):
                    filtered_spans += 1
                    continue
                    
                ents.append({
                    "start": int(s),
                    "end": int(e),
                    "label": lab,
                    "pii": bool(label_is_pii(lab)),
                    "text": span_text
                })
                
            results[uid] = ents

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    
    # Write results
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Processed {len(results)} inputs")
    if total_spans > 0:
        print(f"Filtered out {filtered_spans}/{total_spans} spans "
              f"({filtered_spans/max(1, total_spans)*100:.1f}%)")
    print(f"Wrote predictions to {args.output}")


if __name__ == "__main__":
    main()
