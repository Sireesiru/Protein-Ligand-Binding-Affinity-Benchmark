import os
import numpy as np
import argparse
import torch
import polars as pl
from tqdm import tqdm
#from spectral_utils import compute_ecp

# ------------------------
#   Embedding functions
# ------------------------

def embed(model, tokenizer, strings: list[str]):
    device = next(model.parameters()).device
    
    # ESM2 models handle tokenization differently than ProtGPT2. 
    inputs = tokenizer(strings,
                       return_tensors="pt",
                       padding=True,
                       truncation=True, # Critical for ESM2 (max 1024)
                       max_length=1024).to(device)

    attn_mask = inputs.get("attention_mask", 0)
    
    with torch.no_grad():
        outputs = model(**inputs)

    last_hidden_state = outputs.last_hidden_state

    return attn_mask, last_hidden_state

def mean_pool(attn_mask, last_hidden_state):
    if attn_mask is not None:
        mask = attn_mask.unsqueeze(-1)
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1)
        embeddings = summed / counts
    else:
        embeddings = last_hidden_state.mean(dim=1)
    return embeddings

# ------------------------
#         Main
# ------------------------
def main():
    parser = argparse.ArgumentParser()
    # Tweaked defaults for pdbbind_canonical_affinities.csv
    parser.add_argument("--csv", type=str, default="pdbbind_canonical_affinities.csv")
    parser.add_argument("--col", type=str, default="seq") 
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--outdir", type=str, default="./esm2_embeddings")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--model", type=str, default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--tokenizer", type=str, default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--method", type=str, default="mean", help="['mean' | 'ecp']")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Set GPU
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")

    # Load Dataset - Row order is strictly maintained by Polars
    pdbbind = pl.read_csv(args.csv)
    sequences = pdbbind.get_column(args.col).to_list()
    print(f"Total sequences: {len(sequences)}")
    # DEBUG MODE
    #sequences = sequences[:10]
    #print("Running debug mode on 10 sequences")
    # Set Pooling Method based on collaborator's code
    if args.method == "mean":
        pool_func = mean_pool
    elif args.method == "ecp":
        def pool_func(mask, e) -> torch.Tensor:
            cleaned_emb = []
            for i in range(e.shape[0]):
                valid_len = mask[i].sum().item()
                # Collaborator logic: slice out special tokens [1:-1]
                res_only = e[i, 1:int(valid_len-1), :]
                cleaned_emb.append(compute_ecp(res_only))
            return torch.stack(cleaned_emb)
    else:
        raise ValueError("method unknown. Use ['mean' | 'ecp']")

    # Load Model/Tokenizer from your OLCF shared cache
    from transformers import AutoModel, AutoTokenizer
    hf_cache = "/lustre/orion/gen006/world-shared/booshan/.cache/huggingface/"

    tokenizer_obj = AutoTokenizer.from_pretrained(args.tokenizer, cache_dir=hf_cache, local_files_only=True)
    model_obj = AutoModel.from_pretrained(args.model, 
                                          cache_dir=hf_cache, 
                                          torch_dtype=torch.bfloat16, # Use bfloat16 for Frontier/A100 efficiency
                                          local_files_only=True).to(device).eval()

    # Embedding Generation
    all_embs = []
    for i in tqdm(range(0, len(sequences), args.batch_size), desc=f"ESM2 {args.method}"):
        batch = sequences[i : i + args.batch_size]
        mask, hidden = embed(model_obj, tokenizer_obj, batch)
        emb = pool_func(mask, hidden)
        all_embs.append(emb.cpu().float().numpy())

    # Final Output
    out = np.concatenate(all_embs, axis=0)
    out_path = os.path.join(args.outdir, f"esm2_embeddings_{args.method}.npy")
    np.save(out_path, out)

    print(f"Final matrix saved: {out.shape} -> {out_path}")

if __name__ == "__main__":
    main()