# import random
# import json

# names = ["ramesh sharma", "anita rao", "kiran kumar", "fatima sheikh", "vijay menon", "sneha patel", "arjun das", "priyanka roy"]
# emails = [f"{name.replace(' ', ' dot ')} at gmail dot com" for name in names]
# phones = [
#     "nine eight seven six five four three two one zero",
#     "eight zero zero one two three four five six seven",
#     "nine nine nine eight eight seven seven six six five",
#     "nine one seven zero six zero five zero four zero"
# ]
# credit_cards = [
#     "four two four two four two four two four two four two four two four two",
#     "five five five five five five five five six six six six six six six six",
#     "three seven eight two eight two two four six three one zero zero zero five"
# ]
# dates = ["twenty second january", "fifteenth august", "third march", "twenty fifth december", "first june", "fourteenth april"]
# cities = ["chennai", "mumbai", "delhi", "bangalore", "hyderabad", "pune", "ahmedabad"]
# locations = ["central station", "airport road", "main bazar", "MG road"]

# examples = []
# for i in range(1000): # Train
#     utt_id = f"utt_{i+1:04d}"
#     person = random.choice(names)
#     phone = random.choice(phones)
#     card = random.choice(credit_cards)
#     email = random.choice(emails)
#     date = random.choice(dates)
#     city = random.choice(cities)
#     location = random.choice(locations)
#     template = random.choice([
#         ("my credit card number is {} and my email is {} and my phone number is {}", [card, email, phone], ["CREDIT_CARD", "EMAIL", "PHONE"]),
#         ("call {} in {} tomorrow {}", [person, city, date], ["PERSON_NAME", "CITY", "DATE"]),
#         ("reaching {} on {}", [location, date], ["LOCATION", "DATE"]),
#         ("send details to {} located in {}", [email, city], ["EMAIL", "CITY"]),
#         ("say hello to {} and meet at {}", [person, location], ["PERSON_NAME", "LOCATION"]),
#         ("my card is {} my mobile is {}", [card, phone], ["CREDIT_CARD", "PHONE"]),
#         ("travelling to {} from {}", [city, location], ["CITY", "LOCATION"]),
#         ("email {} for queries", [email], ["EMAIL"]),
#         ("call me at {}", [phone], ["PHONE"]),
#     ])
#     text = template[0].format(*template[1])
#     entities = []
#     for idx, ent_text in enumerate(template[1]):
#         label = template[2][idx]
#         start = text.find(ent_text)
#         if start != -1:
#             end = start + len(ent_text)
#             entities.append({"start": start, "end": end, "label": label})
#     examples.append({"id": utt_id, "text": text, "entities": entities})

# # Write train.jsonl (first 800), dev.jsonl (next 200), test.jsonl (last 100, *no entities*)
# with open("data/train.jsonl", "w") as f:
#     for ex in examples[:800]:
#         f.write(json.dumps(ex) + "\n")
# with open("data/dev.jsonl", "w") as f:
#     for ex in examples[800:1000]:
#         f.write(json.dumps(ex) + "\n")
# with open("data/test.jsonl", "w") as f:
#     for ex in examples[900:1000]:
#         f.write(json.dumps({"id": ex["id"], "text": ex["text"]}) + "\n")



# import random, json, os
# from datetime import datetime, timedelta

# OUTPUT_DIR = "data_diverse_holdout"
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # full pools (expand these lists for realism)
# NAMES = ["rahul sharma","sneha patel","arjun reddy","ravi kumar","anita singh","john doe","mary ann","naveen verma","alok kumar","deepika rajput","vikas joshi","priya singh","sanjay mehta","tara rai","ronit bose"]
# CITIES = ["mumbai","chennai","delhi","kolkata","bengaluru","hyderabad","pune","ahmedabad","jaipur","lucknow"]
# DOMAINS = ["gmail","yahoo","hotmail","outlook","protonmail","icloud","fastmail","example"]
# LOCATIONS = ["near central mall","beside railway station","opposite metro gate","near city hospital","around market area","next to school"]
# # holdout fraction
# HOLDOUT_FRAC = 0.2

# def split_holdout(pool, frac=HOLDOUT_FRAC):
#     pool = pool[:] 
#     random.shuffle(pool)
#     k = max(1, int(len(pool) * frac))
#     return pool[k:], pool[:k]  # train_pool, holdout_pool

# names_train, names_holdout = split_holdout(NAMES)
# cities_train, cities_holdout = split_holdout(CITIES)
# domains_train, domains_holdout = split_holdout(DOMAINS)
# locations_train, locations_holdout = split_holdout(LOCATIONS)

# # fragments for template-free composition
# subj_fragments = ["i need", "please send", "contact", "call", "reach out to", "my friend", "the customer", "she said", "we booked", "booking for"]
# verb_fragments = ["on", "at", "for", "about", "regarding", "to", "with"]
# ending_fragments = ["as soon as possible", "today", "tomorrow", "this week", "next month", "right now", "please"]

# # simple generators
# def gen_phone_spoken():
#     return " ".join(random.choice("0123456789") for _ in range(10))
# def gen_cc_spoken():
#     return " ".join(random.choice("0123456789") for _ in range(16))
# def gen_email_spoken(name, domain):
#     return name.replace(" ", " dot ") + " at " + domain + " dot com"
# def gen_date_spoken():
#     start = datetime(2018,1,1)
#     d = start + timedelta(days=random.randint(0,3000))
#     return d.strftime("%d %B %Y").lower().replace(" 0"," ")

# # paraphrases map (small)
# paraphrases = {
#     "call": ["call", "ring", "give me a call", "phone me"],
#     "email": ["email", "drop an email", "mail", "send an email"],
# }

# def paraphrase_word(word):
#     if word in paraphrases:
#         return random.choice(paraphrases[word])
#     return word

# # Masking: randomly mask parts of entity (for training only)
# def mask_entity(value, label):
#     if label == "PHONE":
#         # keep last 4 digits sometimes
#         if random.random() < 0.4:
#             parts = value.split()
#             return " ".join(["xx"]*(len(parts)-4) + parts[-4:])
#     if label == "CREDIT_CARD":
#         if random.random() < 0.3:
#             parts = value.split()
#             return " ".join(["xx"]*(len(parts)-8) + parts[-8:])
#     if label == "EMAIL":
#         if random.random() < 0.3:
#             # mask local-part partially
#             parts = value.split(" at ")
#             if len(parts) == 2:
#                 local = parts[0]
#                 if "dot" in local:
#                     pieces = local.split(" dot ")
#                     # mask some pieces
#                     for i in range(len(pieces)):
#                         if random.random() < 0.5:
#                             pieces[i] = "xx"
#                     return " dot ".join(pieces) + " at " + parts[1]
#     return value

# # generate a sentence with controlled pools (train_mode True/False)
# def generate_sentence(train_mode=True):
#     # pick whether to use holdout pools or training pools depending on mode
#     names_pool = names_train if train_mode else names_holdout
#     cities_pool = cities_train if train_mode else cities_holdout
#     domains_pool = domains_train if train_mode else domains_holdout
#     locations_pool = locations_train if train_mode else locations_holdout

#     # sample entity types to include (1-3 entity types)
#     ent_types = random.sample(["PERSON_NAME","EMAIL","PHONE","CREDIT_CARD","DATE","CITY","LOCATION"], k=random.randint(1,3))
#     fragments = []
#     entities = []

#     for ent in ent_types:
#         if ent == "PERSON_NAME":
#             val = random.choice(names_pool)
#         elif ent == "EMAIL":
#             nm = random.choice(names_pool)
#             dom = random.choice(domains_pool)
#             val = gen_email_spoken(nm, dom)
#         elif ent == "PHONE":
#             val = gen_phone_spoken()
#         elif ent == "CREDIT_CARD":
#             val = gen_cc_spoken()
#         elif ent == "DATE":
#             val = gen_date_spoken()
#         elif ent == "CITY":
#             val = random.choice(cities_pool)
#         elif ent == "LOCATION":
#             val = random.choice(locations_pool)
#         # possibly mask (applied only for train_mode to force generalization)
#         if train_mode and random.random() < 0.25:
#             val_for_text = mask_entity(val, ent)
#         else:
#             val_for_text = val

#         # build a small fragment containing the entity in random context
#         left = random.choice(subj_fragments + [""])
#         connector = random.choice(verb_fragments + [""])
#         right = random.choice(ending_fragments + [""])
#         # paraphrase call/email words
#         left_par = " ".join(paraphrase_word(w) for w in left.split())
#         frag = " ".join(p for p in [left_par, connector, val_for_text, right] if p).strip()
#         fragments.append(frag)

#         entities.append((ent, val_for_text))  # store exact text used (for later span computing)

#     # shuffle fragments to change order
#     random.shuffle(fragments)
#     sentence = " , ".join(fragments)
#     # add noise: stutter, filler, double spacing probabilistically
#     if random.random() < 0.3:
#         sentence = random.choice(["uh","umm","you know"]) + " " + sentence
#     if random.random() < 0.2:
#         sentence = sentence.replace(" ", "  ")
#     return sentence, entities

# # function to compute spans by searching entity texts in the final noisy sentence
# def compute_spans(sentence, entities):
#     spans = []
#     search_from = 0
#     for label, value in entities:
#         idx = sentence.find(value, search_from)
#         if idx == -1:
#             # fallback: try anywhere
#             idx = sentence.find(value)
#         if idx != -1:
#             spans.append({"start": idx, "end": idx + len(value), "label": label})
#             search_from = idx + len(value)
#     return spans

# # Generate datasets
# def generate_set(n, train_mode=True, start_id=0):
#     out = []
#     for i in range(n):
#         s, ents = generate_sentence(train_mode=train_mode)
#         spans = compute_spans(s, ents)
#         out.append({"id": f"utt_{start_id+i:06d}", "text": s, "entities": spans})
#     return out

# train = generate_set(800, train_mode=True, start_id=0)
# dev   = generate_set(200, train_mode=True, start_id=800)  # dev from train pool
# test  = generate_set(200, train_mode=False, start_id=1000)  # test uses holdout pools

# # Save
# def write_jsonl(path, data):
#     with open(path, "w", encoding="utf-8") as f:
#         for x in data:
#             f.write(json.dumps(x, ensure_ascii=False) + "\n")

# write_jsonl(os.path.join(OUTPUT_DIR, "train.jsonl"), train)
# write_jsonl(os.path.join(OUTPUT_DIR, "dev.jsonl"), dev)
# write_jsonl(os.path.join(OUTPUT_DIR, "test.jsonl"), test)

# print("Saved datasets to", OUTPUT_DIR)



# generate_diverse_dataset.py
# Updated full dataset generator with marker parsing bug fixed (robust regex extraction)
# Produces: train.jsonl, dev_clean.jsonl, dev_noisy.jsonl, test.jsonl
# Run: python generate_diverse_dataset.py

import random
import json
import os
import re
from datetime import datetime, timedelta
from collections import Counter

# ------------------ CONFIG ------------------
OUTPUT_DIR = "data_diverse_holdout_v2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_TRAIN = 1200
NUM_DEV = 200
NUM_TEST = 400

HOLDOUT_FRAC = 0.20    # 20% holdout values used only for test
MAX_ENTITIES_PER_SENT = 3

# Oversampling weights (increase rare classes)
ENTITY_POOL = [
    ("PERSON_NAME", 1.0),
    ("EMAIL", 1.2),
    ("PHONE", 1.0),
    ("CREDIT_CARD", 0.8),
    ("DATE", 1.0),
    ("CITY", 0.6),
    ("LOCATION", 0.6),
]

ENTITY_LABELS = [e for e, _ in ENTITY_POOL]
ENTITY_WEIGHTS = [w for _, w in ENTITY_POOL]

# distractor (hard-negative) probability
DISTRACTOR_PROB = 0.20

# token-level noise probs (training)
FILLER_PROB = 0.25
SPELL_NOISE_PROB = 0.20
DUPLICATE_PROB = 0.15
DROP_WORD_PROB = 0.12
STUTTER_PROB = 0.10
DOUBLE_SPACE_PROB = 0.15

# Masking probabilities (training only)
MASK_PHONE_PROB = 0.40
MASK_CC_PROB = 0.30
MASK_EMAIL_PROB = 0.30

# ------------------ POOLS (extend for realism) ------------------
NAMES = [
    "rahul sharma","sneha patel","arjun reddy","ravi kumar","anita singh",
    "john doe","mary ann","naveen verma","alok kumar","deepika rajput",
    "vikas joshi","priya singh","sanjay mehta","tara rai","ronit bose",
    "kiran nair","meenal patel","amit gupta","richa verma","sudhir naik",
    "alex smith","linda brown","omar khan","leila ahmed"
]

CITIES = [
    "mumbai","chennai","delhi","kolkata","bengaluru","hyderabad",
    "pune","ahmedabad","jaipur","lucknow","surat","vadodara","coimbatore"
]

DOMAINS = ["gmail","yahoo","hotmail","outlook","protonmail","icloud","fastmail","example","gmx"]

LOCATIONS = [
    "near central mall","beside railway station","opposite metro gate",
    "near city hospital","around market area","next to school",
    "behind the old library","in front of the temple"
]

FILLERS = ["uh", "umm", "you know", "so yeah", "basically", "actually", "like"]

# small distractor fragments that look entity-like but should not be labeled
DISTRACTORS = [
    "room number 23", "invoice no 4567", "order id 9988", "table number 5",
    "flight 789", "ticket num 1234", "ref 5555"
]

# fragments for template-free composition
SUBJ_FRAGMENTS = ["i need", "please send", "contact", "call", "reach out to", 
                  "my friend", "the customer", "she said", "we booked", "booking for", "remember"]
VERB_FRAGMENTS = ["on", "at", "for", "about", "regarding", "to", "with"]
ENDING_FRAGMENTS = ["as soon as possible", "today", "tomorrow", "this week", "next month", "right now", "please", "thanks"]

# ------------------ HELPERS ------------------
def split_holdout(pool, frac=HOLDOUT_FRAC):
    p = pool[:]
    random.shuffle(p)
    k = max(1, int(len(p) * frac))
    return p[k:], p[:k]

names_train, names_holdout = split_holdout(NAMES)
cities_train, cities_holdout = split_holdout(CITIES)
domains_train, domains_holdout = split_holdout(DOMAINS)
locations_train, locations_holdout = split_holdout(LOCATIONS)

# Weighted choice of entity types for diversity + oversampling
def sample_entities(k=1):
    return random.choices(ENTITY_LABELS, weights=ENTITY_WEIGHTS, k=k)

# Generators
def gen_phone_spoken():
    return " ".join(random.choice("0123456789") for _ in range(10))

def gen_cc_spoken():
    return " ".join(random.choice("0123456789") for _ in range(16))

def gen_email_spoken(name, domain):
    return name.replace(" ", " dot ") + " at " + domain + " dot com"

def gen_date_spoken():
    start = datetime(2018,1,1)
    d = start + timedelta(days=random.randint(0,3000))
    return d.strftime("%d %B %Y").lower().replace(" 0", " ")

# Masking
def mask_entity(value, label, train_mode=True):
    if not train_mode:
        return value
    if label == "PHONE" and random.random() < MASK_PHONE_PROB:
        parts = value.split()
        # mask left digits, keep last 4
        if len(parts) > 4:
            return " ".join(["xx"]*(len(parts)-4) + parts[-4:])
    if label == "CREDIT_CARD" and random.random() < MASK_CC_PROB:
        parts = value.split()
        if len(parts) > 8:
            return " ".join(["xx"]*(len(parts)-8) + parts[-8:])
    if label == "EMAIL" and random.random() < MASK_EMAIL_PROB:
        # mask local-part pieces partially
        if " at " in value:
            local, rest = value.split(" at ", 1)
            pieces = local.split(" dot ")
            for i in range(len(pieces)):
                if random.random() < 0.5:
                    pieces[i] = "xx"
            return " dot ".join(pieces) + " at " + rest
    return value

# Small spelling noise on single word (do NOT touch entity tokens)
def spelling_noise(word):
    if len(word) <= 3: return word
    if random.random() < 0.25:
        i = random.randint(1, len(word)-2)
        return word[:i] + word[i+1:]
    if random.random() < 0.25:
        i = random.randint(1, len(word)-2)
        return word[:i] + random.choice("abcdefghijklmnopqrstuvwxyz") + word[i:]
    return word

# Apply noise to non-entity tokens only
def apply_noise_to_tokens(tokens, entity_token_indices):
    out = []
    for i, tok in enumerate(tokens):
        if i in entity_token_indices:
            out.append(tok)  # keep entity token intact
            continue
        # filler insertion
        if random.random() < FILLER_PROB and random.random() < 0.25:
            out.append(random.choice(FILLERS))
        # drop word
        if random.random() < DROP_WORD_PROB:
            continue
        # duplicate
        if random.random() < DUPLICATE_PROB:
            out.append(tok)
        # stutter
        if random.random() < STUTTER_PROB and len(tok) > 2:
            out.append(tok + " " + tok)
        # spelling noise
        if random.random() < SPELL_NOISE_PROB:
            out.append(spelling_noise(tok))
        else:
            out.append(tok)
    # double space possibility
    sentence = " ".join(out)
    if random.random() < DOUBLE_SPACE_PROB:
        sentence = sentence.replace(" ", "  ")
    return sentence

# Build a fragment list where entity values are single tokens (so noise won't corrupt them)
def build_fragment(left, connector, entity_token, right):
    fragment_tokens = []
    if left:
        fragment_tokens.extend(left.split())
    if connector:
        fragment_tokens.extend(connector.split())
    # entity_token is a single element (may contain spaces) — keep as marker
    fragment_tokens.append(entity_token)
    if right:
        fragment_tokens.extend(right.split())
    return fragment_tokens

# Insert distractors occasionally
def maybe_add_distractor(tokens):
    if random.random() < DISTRACTOR_PROB:
        tokens.append(random.choice(DISTRACTORS))
    return tokens

# Compute spans by searching the exact entity substrings in the final text
def compute_spans_from_values(text, entity_values_labels):
    spans = []
    used_positions = []
    for val, label in entity_values_labels:
        # find the first occurrence of val that doesn't overlap previous matches
        start = 0
        found = -1
        while True:
            idx = text.find(val, start)
            if idx == -1:
                break
            end = idx + len(val)
            # check overlap
            overlaps = any(not (end <= a or idx >= b) for (a,b,_) in used_positions)
            if not overlaps:
                found = idx
                used_positions.append((idx, end, label))
                break
            start = idx + 1
        if found != -1:
            spans.append({"start": found, "end": found + len(val), "label": label})
    # sort spans by start
    spans.sort(key=lambda x: x["start"])
    return spans

# ------------------ GENERATION ------------------
def generate_example(example_id, train_mode=True):
    # choose number of entities
    num_entities = random.randint(1, MAX_ENTITIES_PER_SENT)
    ent_types = sample_entities(k=num_entities)

    fragments_tokens = []  # list of token lists; entity tokens will be single special tokens
    entity_values_labels = []  # (value string inserted into tokens, label)

    for ent in ent_types:
        # pick pools depending on mode
        if ent == "PERSON_NAME":
            pool = names_train if train_mode else names_holdout
            val = random.choice(pool)
        elif ent == "EMAIL":
            pool = names_train if train_mode else names_holdout
            dom_pool = domains_train if train_mode else domains_holdout
            name_val = random.choice(pool)
            dom_val = random.choice(dom_pool)
            val = gen_email_spoken(name_val, dom_val)
        elif ent == "PHONE":
            val = gen_phone_spoken()
        elif ent == "CREDIT_CARD":
            val = gen_cc_spoken()
        elif ent == "DATE":
            val = gen_date_spoken()
        elif ent == "CITY":
            pool = cities_train if train_mode else cities_holdout
            val = random.choice(pool)
        elif ent == "LOCATION":
            pool = locations_train if train_mode else locations_holdout
            val = random.choice(pool)
        else:
            val = "unknown"

        # apply masking (affects the actual string we must find later)
        val_for_text = mask_entity(val, ent, train_mode=train_mode)

        # build a small fragment with entity token as single element
        left = random.choice(SUBJ_FRAGMENTS + [""])
        connector = random.choice(VERB_FRAGMENTS + [""])
        right = random.choice(ENDING_FRAGMENTS + [""])

        # entity token: use a unique marker that we will replace by the real value when joining tokens
        entity_marker = f"__ENT_{len(entity_values_labels)}__"
        fragment_token_list = build_fragment(left, connector, entity_marker, right)
        # record mapping marker -> real value & label
        entity_values_labels.append((val_for_text, ent))

        # append fragment token list to global tokens
        fragments_tokens.append(fragment_token_list)

    # Optionally add distractor fragment (hard negative) — not added to entity_values_labels
    if random.random() < DISTRACTOR_PROB:
        fragments_tokens.append([random.choice(DISTRACTORS)])

    # shuffle fragments order to reduce templating
    random.shuffle(fragments_tokens)

    # Flatten tokens but keep track of where entity markers are
    flat_tokens = []
    marker_to_value = {}
    for frag in fragments_tokens:
        for tok in frag:
            if tok.startswith("__ENT_") and tok.endswith("__"):
                # extract numeric index robustly using regex
                m = re.match(r"__ENT_(\d+)__", tok)
                if m:
                    marker_idx = int(m.group(1))
                else:
                    # fallback to safer replace method
                    try:
                        marker_idx = int(tok.replace("__ENT_", "").replace("__", ""))
                    except Exception:
                        raise ValueError(f"Bad entity marker format: {tok}")
                flat_tokens.append(tok)
                # guard marker_idx range
                if marker_idx < 0 or marker_idx >= len(entity_values_labels):
                    raise IndexError(f"Marker index {marker_idx} out of range (entities={len(entity_values_labels)})")
                marker_to_value[tok] = entity_values_labels[marker_idx][0]
            else:
                flat_tokens.append(tok)
        # add separator comma token sometimes to appear conversational
        if random.random() < 0.4:
            flat_tokens.append(",")

    # possible filler at start (training only)
    if train_mode and random.random() < 0.25:
        flat_tokens.insert(0, random.choice(FILLERS))

    # Decide which token indices are entities so noise won't edit them
    entity_token_indices = set(i for i, tok in enumerate(flat_tokens) if tok.startswith("__ENT_"))

    # Apply noise only to non-entity tokens and produce sentence where markers remain
    noisy_sentence_with_markers = apply_noise_to_tokens(flat_tokens, entity_token_indices)

    # Replace markers by the real entity strings (which themselves may contain spaces)
    final_text = noisy_sentence_with_markers
    # ensure markers are replaced in a safe order (longest first to avoid partial collisions)
    for marker in sorted(marker_to_value.keys(), key=len, reverse=True):
        final_text = final_text.replace(marker, marker_to_value[marker])

    # Compute spans by searching values in final text (guaranteed to find as we inserted exact strings)
    spans = compute_spans_from_values(final_text, entity_values_labels)

    return {"id": f"utt_{example_id:06d}", "text": final_text, "entities": spans}

# Generate sets
def generate_set(n, train_mode=True, start_id=0):
    out = []
    for i in range(n):
        ex = generate_example(start_id + i, train_mode=train_mode)
        out.append(ex)
    return out

# Create datasets
train = generate_set(NUM_TRAIN, train_mode=True, start_id=0)
dev_clean = generate_set(NUM_DEV//2, train_mode=False, start_id=NUM_TRAIN)
# For dev_noisy we generate using train pools but with noise (training-mode) to simulate noisy dev
dev_noisy = generate_set(NUM_DEV - NUM_DEV//2, train_mode=True, start_id=NUM_TRAIN + NUM_DEV//2)
test = generate_set(NUM_TEST, train_mode=False, start_id=NUM_TRAIN + NUM_DEV)

# Save to JSONL
def write_jsonl(path, data):
    with open(path, "w", encoding="utf-8") as f:
        for x in data:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

write_jsonl(os.path.join(OUTPUT_DIR, "train.jsonl"), train)
write_jsonl(os.path.join(OUTPUT_DIR, "dev_clean.jsonl"), dev_clean)
write_jsonl(os.path.join(OUTPUT_DIR, "dev_noisy.jsonl"), dev_noisy)
write_jsonl(os.path.join(OUTPUT_DIR, "test.jsonl"), test)

# Simple overlap diagnostics
def collect_entity_texts(dataset):
    texts = []
    for ex in dataset:
        for e in ex["entities"]:
            texts.append(ex["text"][e["start"]:e["end"]])
    return set(texts)

train_texts = collect_entity_texts(train)
test_texts = collect_entity_texts(test)
overlap = len(train_texts & test_texts)
print(f"Saved datasets to {OUTPUT_DIR}")
print(f"Train examples: {len(train)}, Dev_clean: {len(dev_clean)}, Dev_noisy: {len(dev_noisy)}, Test: {len(test)}")
print(f"Unique entity strings in train: {len(train_texts)}, unique in test: {len(test_texts)}, overlap: {overlap}")
