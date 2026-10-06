import torch
import numpy as np
import pandas as pd
from transformers import GPT2Config, GPT2Tokenizer, GPT2LMHeadModel
from tqdm import tqdm
import os

# ---------------- CONFIG ----------------
MODEL_DIR = "./local_protgpt2"
CSV_FILE = "pdbbind_canonical_affinities.csv" 
OUT_NPY = "pdbbind_protgpt2_embeddings_FINAL.npy"
OUT_META = "pdbbind_protgpt2_metadata_FINAL.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 1024
# ----------------------------------------

print(f"Loading Master CSV: {CSV_FILE}")
df = pd.read_csv(CSV_FILE)
df['seq'] = df['seq'].fillna("EMPTY").astype(str).str.strip()

print(f"Loading ProtGPT2 using explicit GPT2 classes...")

# 1. Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)

# 2. Load the model using the SPECIFIC GPT2 class 
# This ignores the 'architectures' registry in config.json
model = GPT2LMHeadModel.from_pretrained(
    MODEL_DIR, 
    output_hidden_states=True
).to(DEVICE)
model.eval()

all_embeddings = []

print(f"Processing {len(df)} rows linearly to ensure 1:1 alignment...")

with torch.no_grad():
    for i, row in tqdm(df.iterrows(), total=len(df)):
        seq = row['seq']
        
        if seq == "EMPTY" or len(seq) < 2:
            emb = np.zeros((1280,)) # Matches n_embd in your config
        else:
            # ProtGPT2 prefix
            inputs = tokenizer("<|endoftext|>" + seq, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(DEVICE)
            outputs = model(**inputs)
            
            # hidden_states[-1] is the last layer
            # Shape: (1, Sequence_Length, 1280)
            hidden = outputs.hidden_states[-1].squeeze(0)
            
            # Mean pooling over residues
            emb = hidden.mean(dim=0).cpu().numpy()
        
        all_embeddings.append(emb)

        if (i + 1) % 100 == 0:
            torch.cuda.empty_cache()

# ---------------- SAVE ----------------
print("\nStacking and Saving...")
final_matrix = np.stack(all_embeddings)
np.save(OUT_NPY, final_matrix)

# Save the receipt to prove order matches exactly
df[['pdb_id', 'seq', 'neg_log10_affinity_M']].to_csv(OUT_META, index=False)

print(f"\n✅ SUCCESS! {OUT_NPY} is now 1:1 with {CSV_FILE}")