#import os
#import sys
#import numpy as np
#import argparse
#import torch
#import polars as pl
#from tqdm import tqdm
#
## ------------------------
## Load model
## ------------------------
#def load_model(model_path, device):
#    sys.path.append(os.path.dirname(model_path))
#
#    from progen2_model.modeling_progen import ProGenForCausalLM
#    from transformers import PreTrainedTokenizerFast
#
#    tokenizer = PreTrainedTokenizerFast(
#        tokenizer_file=f"{model_path}/tokenizer.json"
#    )
#
#    model = ProGenForCausalLM.from_pretrained(
#        model_path,
#        torch_dtype=torch.bfloat16
#    ).to(device).eval()
#
#    return model, tokenizer
#
#
## ------------------------
## Embed ONE sequence (no padding!)
## ------------------------
#def embed_sequence(model, tokenizer, seq, device):
#    inputs = tokenizer(
#        seq,
#        return_tensors="pt",
#        truncation=True,
#        max_length=1024
#    ).to(device)
#
#    with torch.no_grad():
#        outputs = model(**inputs, output_hidden_states=True)
#
#    hidden = outputs.hidden_states[-1]  # [1, L, D]
#
#    # mean pooling over sequence length
#    emb = hidden[0].mean(dim=0)
#
#    #return emb.cpu().numpy()
#    return emb.detach().cpu().float().numpy()
#
#
## ------------------------
## Main
## ------------------------
#def main():
#    parser = argparse.ArgumentParser()
#    parser.add_argument("--csv", default="pdbbind_canonical_affinities.csv")
#    parser.add_argument("--col", default="seq")
#    parser.add_argument("--outdir", default="./progen2_embeddings")
#    parser.add_argument("--gpu", type=int, default=0)
#    parser.add_argument("--debug_n", type=int, default=None)
#
#    args = parser.parse_args()
#
#    os.makedirs(args.outdir, exist_ok=True)
#
#    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
#
#    # ------------------------
#    # Load data (ORDER PRESERVED)
#    # ------------------------
#    df = pl.read_csv(args.csv)
#    sequences = df.get_column(args.col).to_list()
#
#    print(f"Total sequences: {len(sequences)}")
#
#    if args.debug_n:
#        sequences = sequences[:args.debug_n]
#        print(f"DEBUG MODE: {len(sequences)} sequences")
#
#    # ------------------------
#    # Load model
#    # ------------------------
#    model_path = "/lustre/orion/gen006/scratch/sireesiru/miniffinity/progen2_model"
#    model, tokenizer = load_model(model_path, device)
#
#    # ------------------------
#    # Embedding loop (ORDER PRESERVED)
#    # ------------------------
#    embeddings = []
#
#    for seq in tqdm(sequences, desc="ProGen2 (no padding)"):
#        emb = embed_sequence(model, tokenizer, seq, device)
#        embeddings.append(emb)
#
#    embeddings = np.vstack(embeddings)
#
#    out_path = os.path.join(args.outdir, "progen2_embeddings.npy")
#    np.save(out_path, embeddings)
#
#    print(f"\nSaved: {embeddings.shape} -> {out_path}")
#
#
#if __name__ == "__main__":
#    main()

import os
import sys
import numpy as np
import argparse
import torch
import polars as pl
from tqdm import tqdm

# ------------------------
# Load model
# ------------------------
def load_model(model_path, device):
    sys.path.append(os.path.dirname(model_path))

    from progen2_model.modeling_progen import ProGenForCausalLM
    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=f"{model_path}/tokenizer.json"
    )

    model = ProGenForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16
    ).to(device).eval()

    return model, tokenizer


# ------------------------
# Embed one sequence (NO padding)
# ------------------------
def embed_sequence(model, tokenizer, seq, device):
    inputs = tokenizer(
        seq,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden = outputs.hidden_states[-1]
    emb = hidden[0].mean(dim=0)

    return emb.detach().cpu().float().numpy()


# ------------------------
# Main
# ------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv", default="pdbbind_canonical_affinities.csv")
    parser.add_argument("--col", default="seq")
    parser.add_argument("--outdir", default="./progen2_embeddings")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--chunk", type=int, default=0)
    parser.add_argument("--num_chunks", type=int, default=1)

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # ------------------------
    # Load data (ORDER PRESERVED)
    # ------------------------
    df = pl.read_csv(args.csv)
    sequences = df.get_column(args.col).to_list()

    n = len(sequences)

    # ------------------------
    # Chunk slicing
    # ------------------------
    chunk_size = (n + args.num_chunks - 1) // args.num_chunks
    start = args.chunk * chunk_size
    end = min(start + chunk_size, n)

    sequences = sequences[start:end]

    print(f"Processing chunk {args.chunk}: {start} → {end}")

    # ------------------------
    # Load model
    # ------------------------
    model_path = "/lustre/orion/gen006/scratch/sireesiru/miniffinity/progen2_model"
    model, tokenizer = load_model(model_path, device)

    # ------------------------
    # Embedding loop
    # ------------------------
    embeddings = []

    for seq in tqdm(sequences, desc=f"Chunk {args.chunk}"):
        emb = embed_sequence(model, tokenizer, seq, device)
        embeddings.append(emb)

    embeddings = np.vstack(embeddings)

    out_path = os.path.join(args.outdir, f"progen2_chunk_{args.chunk}.npy")
    np.save(out_path, embeddings)

    print(f"Saved chunk {args.chunk}: {embeddings.shape}")


if __name__ == "__main__":
    main()