# -----------------------------------------------------------
# 0. Imports & helper functions
# -----------------------------------------------------------
import torch, random, heapq, json, os
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from scipy.stats import pearsonr
import openai   # pip install openai>=1.3.1

DEVICE = "cuda:0"
TOKENIZER = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")  # any GPT‑2‑style tokenizer
openai.api_key = os.environ["OPENAI_API_KEY"]

WINDOW = 64                        # length of text fragments
TOP_K   = 20                       # how many top‑activating examples to keep
MIN_COUNT = 20                     # ≥20 tokens with activation>0 required
TARGET_FEATURES = 150              # evaluate at most 150 features

def tokens_to_text(tok):
    # tok: [T] or [B,T] tensor of token ids (assume GPT‑2 bpe)
    if tok.ndim == 2: tok = tok.flatten()
    return TOKENIZER.decode(tok.cpu().tolist())

def call_gpt4(prompt, **kw):
    resp = openai.ChatCompletion.create(**{
        "model": "gpt-4o-mini",      # or gpt-4o when available
        "messages": [{"role":"user","content":prompt}],
        **kw
    })
    return resp.choices[0].message.content.strip()

def call_gpt35(prompt, **kw):
    resp = openai.ChatCompletion.create(**{
        "model": "gpt-3.5-turbo-0125",
        "messages":[{"role":"user","content":prompt}],
        **kw
    })
    return resp.choices[0].message.content.strip()

# -----------------------------------------------------------
# 1. First pass – gather activation statistics & top windows
# -----------------------------------------------------------
feature_counts  = None          # will be [n_features] tensor
feature_top     = []            # length = n_features; each is a min‑heap of (act, window_tokens, per_token_acts)

for batch_idx, batch in tqdm(enumerate(eval_loader),
                             total=n_batches, desc="Eval‑Pass‑1"):
    if n_batches is not None and batch_idx >= n_batches:
        break

    tokens = batch[eval_config.column_name].to(DEVICE)    # [B,S]
    new_logits, new_acts = model.forward(
        tokens=tokens,
        sae_positions=model.raw_sae_positions,  # <- from your snippet
        cache_positions=None,
    )
    coeffs = new_acts['blocks.4.hook_resid_pre'].c        # [B, S, N_feat]
    B, S, N = coeffs.shape

    # lazy init
    if feature_counts is None:
        feature_counts = torch.zeros(N, dtype=torch.long, device=DEVICE)
        feature_top    = [ [] for _ in range(N) ]

    # count non‑zero activations per feature
    feature_counts += (coeffs > 0).sum(dim=(0,1)).to(torch.long)

    # maintain top‑K windows per feature
    for b in range(B):
        # for start in range(0, S, WINDOW):
        #     end = min(start+WINDOW, S)
        #     window_coef = coeffs[b,start:end]            # [W, N]  (W<=64)
        #     if window_coef.max() == 0: 
        #         continue  # skip windows with no activation at all
            
        _coeffs = coeffs[b]
        if _coeffs.max() == 0:
            continue

        # we need per‑token activations for this window
        window_tokens = tokens[b]
        window_acts   = _coeffs.cpu()            # move small slice to CPU
        # iterate features with any activation in this window
        for f_idx in (window_acts.max(dim=0).indices[window_acts.max(dim=0).values>0].tolist()):
            max_act = window_acts[:,f_idx].max().item()
            heap = feature_top[f_idx]
            if len(heap) < TOP_K:
                heapq.heappush(heap,(max_act, window_tokens.cpu(), window_acts[:,f_idx]))
            elif max_act > heap[0][0]:
                heapq.heapreplace(heap,(max_act, window_tokens.cpu(), window_acts[:,f_idx]))

# -----------------------------------------------------------
# 2. Choose the 150 features to evaluate
# -----------------------------------------------------------
active_features = (feature_counts >= MIN_COUNT).nonzero(as_tuple=False).squeeze(-1).tolist()
selected = active_features[:TARGET_FEATURES]            # take first 150 (could also random.sample)

print(f"{len(selected)} / {N} features meet the >= {MIN_COUNT} token criterion; "
      f"scoring the first {len(selected)} of them.")

# -----------------------------------------------------------
# 3. For each selected feature – autointerpretation
# -----------------------------------------------------------
results = {}   # feature_idx -> dict(description, score)

for f_idx in tqdm(selected, desc="Autointerpret"):
    # ------------------------------------------------------------------
    # 3a. retrieve TOP_K windows, sort descending by activation
    # ------------------------------------------------------------------
    heap = feature_top[f_idx]
    if len(heap)==0:
        continue   # should not happen if count>=MIN_COUNT
    tops = sorted(heap, key=lambda x: -x[0])             # highest act first

    # pick 5 windows to show GPT‑4    (can use the top‑5)
    demo_windows = tops[:5]

    # build a prompt with color‑coded activations (0‑10) for GPT‑4
    def window_to_text(tok, acts):
        # acts: [W] tensor
        scaled = torch.clamp((acts / acts.max())*10, 0, 10).round().int().tolist()
        tokens = TOKENIZER.convert_ids_to_tokens(tok.tolist(), skip_special_tokens=False)
        return " ".join(f"{t}({a})" for t,a in zip(tokens,scaled))

    prompt_ex = "\n\n".join(window_to_text(t, a) for _,t,a in demo_windows)
    prompt_gpt4 = (
        "Below are excerpts of text. Each token is annotated with a score 0‑10 "
        "indicating how strongly an unknown neuron fires on that token. "
        "Write a short English description of the pattern that makes this neuron fire.\n\n"
        f"{prompt_ex}\n\n"
        "Explanation:"
    )
    explanation = call_gpt4(prompt_gpt4, temperature=0.2, max_tokens=150)

    # ------------------------------------------------------------------
    # 3b. Simulation by GPT‑3.5 on held‑out 10 windows (5 top+5 random)
    # ------------------------------------------------------------------
    random_windows = random.sample(tops[5:], k=min(5, len(tops)-5)) if len(tops)>5 else []
    other_sample   = random.sample(heap, k=min(5, len(heap)))       # independent random
    eval_windows   = (tops[5:10] + random_windows + other_sample)[:10]

    eval_texts = [window_to_text(t,a) for _,t,a in eval_windows]
    eval_true  = [a.tolist() for _,_,a in eval_windows]

    sim_prompt = (
        f"You are given an English explanation of a neuron:\n\"{explanation}\"\n\n"
        "Below are 64‑token snippets. For each token, output an integer 0‑10 "
        "predicting the neuron's activation. Use JSON with one list per snippet.\n\n"
        + "\n\n".join(f"Snippet {i+1}:\n{txt}" for i,txt in enumerate(eval_texts))
    )
    sim_raw = call_gpt35(sim_prompt, temperature=0.0, max_tokens=400)
    try:
        sim_preds = json.loads(sim_raw)
    except json.JSONDecodeError:
        # fallback: try to strip markdown etc.
        sim_raw = sim_raw[sim_raw.find('['): sim_raw.rfind(']')+1]
        sim_preds = json.loads(sim_raw)

    # flatten true & pred, compute Pearson r
    true_flat, pred_flat = [], []
    for t,p in zip(eval_true, sim_preds):
        m = max(t) or 1  # guard div0
        scaled_true = [round((x/m)*10) for x in t]       # same 0‑10 scale
        true_flat.extend(scaled_true)
        pred_flat.extend(p[:len(t)])                     # clip / pad not handled

    score = pearsonr(true_flat, pred_flat)[0] if len(set(pred_flat))>1 else 0.0

    results[f_idx] = dict(description=explanation, score=score)

# -----------------------------------------------------------
# 4. Show / save results
# -----------------------------------------------------------
print("\nTop 10 features by autointerpretability score:")
for f_idx, info in sorted(results.items(), key=lambda x: -x[1]['score'])[:10]:
    print(f"Feature {f_idx:4d}: r = {info['score']:.3f}   {info['description'][:90]}")

with open("autointerpret_results_layer4.json","w") as f:
    json.dump(results, f, indent=2)
