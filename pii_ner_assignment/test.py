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



import random, json, os
from datetime import datetime, timedelta

OUTPUT_DIR = "data_diverse_holdout"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# full pools (expand these lists for realism)
NAMES = ["rahul sharma","sneha patel","arjun reddy","ravi kumar","anita singh","john doe","mary ann","naveen verma","alok kumar","deepika rajput","vikas joshi","priya singh","sanjay mehta","tara rai","ronit bose"]
CITIES = ["mumbai","chennai","delhi","kolkata","bengaluru","hyderabad","pune","ahmedabad","jaipur","lucknow"]
DOMAINS = ["gmail","yahoo","hotmail","outlook","protonmail","icloud","fastmail","example"]
LOCATIONS = ["near central mall","beside railway station","opposite metro gate","near city hospital","around market area","next to school"]
# holdout fraction
HOLDOUT_FRAC = 0.2

def split_holdout(pool, frac=HOLDOUT_FRAC):
    pool = pool[:] 
    random.shuffle(pool)
    k = max(1, int(len(pool) * frac))
    return pool[k:], pool[:k]  # train_pool, holdout_pool

names_train, names_holdout = split_holdout(NAMES)
cities_train, cities_holdout = split_holdout(CITIES)
domains_train, domains_holdout = split_holdout(DOMAINS)
locations_train, locations_holdout = split_holdout(LOCATIONS)

# fragments for template-free composition
subj_fragments = ["i need", "please send", "contact", "call", "reach out to", "my friend", "the customer", "she said", "we booked", "booking for"]
verb_fragments = ["on", "at", "for", "about", "regarding", "to", "with"]
ending_fragments = ["as soon as possible", "today", "tomorrow", "this week", "next month", "right now", "please"]

# simple generators
def gen_phone_spoken():
    return " ".join(random.choice("0123456789") for _ in range(10))
def gen_cc_spoken():
    return " ".join(random.choice("0123456789") for _ in range(16))
def gen_email_spoken(name, domain):
    return name.replace(" ", " dot ") + " at " + domain + " dot com"
def gen_date_spoken():
    start = datetime(2018,1,1)
    d = start + timedelta(days=random.randint(0,3000))
    return d.strftime("%d %B %Y").lower().replace(" 0"," ")

# paraphrases map (small)
paraphrases = {
    "call": ["call", "ring", "give me a call", "phone me"],
    "email": ["email", "drop an email", "mail", "send an email"],
}

def paraphrase_word(word):
    if word in paraphrases:
        return random.choice(paraphrases[word])
    return word

# Masking: randomly mask parts of entity (for training only)
def mask_entity(value, label):
    if label == "PHONE":
        # keep last 4 digits sometimes
        if random.random() < 0.4:
            parts = value.split()
            return " ".join(["xx"]*(len(parts)-4) + parts[-4:])
    if label == "CREDIT_CARD":
        if random.random() < 0.3:
            parts = value.split()
            return " ".join(["xx"]*(len(parts)-8) + parts[-8:])
    if label == "EMAIL":
        if random.random() < 0.3:
            # mask local-part partially
            parts = value.split(" at ")
            if len(parts) == 2:
                local = parts[0]
                if "dot" in local:
                    pieces = local.split(" dot ")
                    # mask some pieces
                    for i in range(len(pieces)):
                        if random.random() < 0.5:
                            pieces[i] = "xx"
                    return " dot ".join(pieces) + " at " + parts[1]
    return value

# generate a sentence with controlled pools (train_mode True/False)
def generate_sentence(train_mode=True):
    # pick whether to use holdout pools or training pools depending on mode
    names_pool = names_train if train_mode else names_holdout
    cities_pool = cities_train if train_mode else cities_holdout
    domains_pool = domains_train if train_mode else domains_holdout
    locations_pool = locations_train if train_mode else locations_holdout

    # sample entity types to include (1-3 entity types)
    ent_types = random.sample(["PERSON_NAME","EMAIL","PHONE","CREDIT_CARD","DATE","CITY","LOCATION"], k=random.randint(1,3))
    fragments = []
    entities = []

    for ent in ent_types:
        if ent == "PERSON_NAME":
            val = random.choice(names_pool)
        elif ent == "EMAIL":
            nm = random.choice(names_pool)
            dom = random.choice(domains_pool)
            val = gen_email_spoken(nm, dom)
        elif ent == "PHONE":
            val = gen_phone_spoken()
        elif ent == "CREDIT_CARD":
            val = gen_cc_spoken()
        elif ent == "DATE":
            val = gen_date_spoken()
        elif ent == "CITY":
            val = random.choice(cities_pool)
        elif ent == "LOCATION":
            val = random.choice(locations_pool)
        # possibly mask (applied only for train_mode to force generalization)
        if train_mode and random.random() < 0.25:
            val_for_text = mask_entity(val, ent)
        else:
            val_for_text = val

        # build a small fragment containing the entity in random context
        left = random.choice(subj_fragments + [""])
        connector = random.choice(verb_fragments + [""])
        right = random.choice(ending_fragments + [""])
        # paraphrase call/email words
        left_par = " ".join(paraphrase_word(w) for w in left.split())
        frag = " ".join(p for p in [left_par, connector, val_for_text, right] if p).strip()
        fragments.append(frag)

        entities.append((ent, val_for_text))  # store exact text used (for later span computing)

    # shuffle fragments to change order
    random.shuffle(fragments)
    sentence = " , ".join(fragments)
    # add noise: stutter, filler, double spacing probabilistically
    if random.random() < 0.3:
        sentence = random.choice(["uh","umm","you know"]) + " " + sentence
    if random.random() < 0.2:
        sentence = sentence.replace(" ", "  ")
    return sentence, entities

# function to compute spans by searching entity texts in the final noisy sentence
def compute_spans(sentence, entities):
    spans = []
    search_from = 0
    for label, value in entities:
        idx = sentence.find(value, search_from)
        if idx == -1:
            # fallback: try anywhere
            idx = sentence.find(value)
        if idx != -1:
            spans.append({"start": idx, "end": idx + len(value), "label": label})
            search_from = idx + len(value)
    return spans

# Generate datasets
def generate_set(n, train_mode=True, start_id=0):
    out = []
    for i in range(n):
        s, ents = generate_sentence(train_mode=train_mode)
        spans = compute_spans(s, ents)
        out.append({"id": f"utt_{start_id+i:06d}", "text": s, "entities": spans})
    return out

train = generate_set(800, train_mode=True, start_id=0)
dev   = generate_set(200, train_mode=True, start_id=800)  # dev from train pool
test  = generate_set(200, train_mode=False, start_id=1000)  # test uses holdout pools

# Save
def write_jsonl(path, data):
    with open(path, "w", encoding="utf-8") as f:
        for x in data:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

write_jsonl(os.path.join(OUTPUT_DIR, "train.jsonl"), train)
write_jsonl(os.path.join(OUTPUT_DIR, "dev.jsonl"), dev)
write_jsonl(os.path.join(OUTPUT_DIR, "test.jsonl"), test)

print("Saved datasets to", OUTPUT_DIR)
