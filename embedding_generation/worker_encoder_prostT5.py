import os
import numpy as np
import argparse
import torch
import polars as pl
from tqdm import tqdm

# ------------------------
#   Embedding functions
# ------------------------

def embed(model, tokenizer, strings):
    device = next(model.parameters()).device

    inputs = tokenizer(strings,
                       return_tensors="pt",
                       padding=True,
                       truncation=False).to(device)

    attn_mask = inputs.get("attention_mask", None)

    with torch.no_grad():
        outputs = model.encoder(**inputs)   

    return attn_mask, outputs.last_hidden_state


def mean_pool(attn_mask, last_hidden_state):
    embeddings = []
    for i in range(last_hidden_state.shape[0]):
        if attn_mask is not None:
            valid_len = attn_mask[i].sum().item()
            res_only = last_hidden_state[i, 1:int(valid_len-1), :]
        else:
            res_only = last_hidden_state[i]
        embeddings.append(res_only.mean(dim=0))
    return torch.stack(embeddings)

# ------------------------
#       Main
# ------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv", type=str, default="pdbbind_canonical_affinities.csv")
    parser.add_argument("--col", type=str, default="seq")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--outdir", type=str, default="./prostt5_embeddings")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--debug_n", type=int, default=None)

    # ProstT5 defaults
    parser.add_argument("--model", type=str, default="Rostlab/prot_t5_xl_uniref50")
    parser.add_argument("--tokenizer", type=str, default="Rostlab/prot_t5_xl_uniref50")

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # ------------------------
    # GPU
    # ------------------------
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # ------------------------
    # Load data (ORDER PRESERVED)
    # ------------------------
    pdbbind = pl.read_csv(args.csv)
    sequences = pdbbind.get_column(args.col).to_list()
    print(f"Total sequences: {len(sequences)}")
    
    if args.debug_n:
        sequences = sequences[:args.debug_n]
        print(f"Running debug mode on {args.debug_n} sequences")
        
    # ProstT5 requires spaced amino acids
    sequences = [" ".join(list(seq)) for seq in sequences]
    print(sequences[0][:50])

    # ------------------------
    # Load model
    # ------------------------
    from transformers import AutoModel, AutoTokenizer

    model_path = "/lustre/orion/gen006/world-shared/booshan/.cache/huggingface/models--Rostlab--ProstT5/snapshots/d7d097d5bf9a993ab8f68488b4681d6ca70db9e5"
    tokenizer = AutoTokenizer.from_pretrained(model_path,use_fast=False,local_files_only=True)
    model = AutoModel.from_pretrained(model_path,torch_dtype=torch.bfloat16,local_files_only=True).to(device).eval()
    
    # ------------------------
    # Embedding loop (ORDER PRESERVED)
    # ------------------------
    all_embs = []

    for i in tqdm(range(0, len(sequences), args.batch_size), desc="ProstT5 mean"):
        batch = sequences[i:i + args.batch_size]
        mask, hidden = embed(model, tokenizer, batch)
        emb = mean_pool(mask, hidden)
        all_embs.append(emb.cpu().float().numpy())

    out = np.concatenate(all_embs, axis=0)

    out_path = os.path.join(args.outdir, "prostt5_embeddings_mean.npy")
    np.save(out_path, out)

    print(f"Final matrix saved: {out.shape} -> {out_path}")


if __name__ == "__main__":
    main()