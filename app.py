import os, json, time, re
import gradio as gr

# Local previews use the existing Space API. Hosted Spaces retain local inference.
BACKEND = os.environ.get("AKSARA_BACKEND", "local" if os.environ.get("SPACE_ID") else "space")
if BACKEND not in {"local", "space"}:
    raise ValueError("AKSARA_BACKEND must be 'local' or 'space'")
if BACKEND == "local":
    import numpy as np
    import faiss
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

# ---------- Paths inside the Space ----------
ART_DIR     = "artifacts"
INDEX_PATH  = os.path.join(ART_DIR, "faiss.index")
PASSAGES    = os.path.join(ART_DIR, "passages.jsonl")
RETR_PATH   = os.path.join(ART_DIR, "retriever_model")       # or "yourname/peribahasa-retriever"
LORA_DIR    = os.path.join(ART_DIR, "qwen-peribahasa-lora")  # or "yourname/qwen-peribahasa-lora"
BASE_MODEL  = "Qwen/Qwen2.5-1.5B-Instruct"

# ---------- Helpers ----------
def load_passages(pfile):
    items = []
    with open(pfile, "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line)["text"])
    return items

def proverb_names_from_passages(pfile, limit=300):
    names, seen = [], set()
    with open(pfile, "r", encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)["text"]
            if " — " in t:
                name = t.split(" — ", 1)[0].strip()
                if name and name.lower() not in seen:
                    seen.add(name.lower())
                    names.append(name)
                    if len(names) >= limit: break
    pin = ["Makan puji", "Makan puluk", "Makan pokok", "Makan ransom", "Pujaan"]
    return [n for n in pin if n in names] + [n for n in names if n not in pin]

def try_extract_exact_meaning(passages, question):
    m = re.search(r'peribahasa\s+["“]?([^"?]+)["”]?', question, flags=re.I)
    target = m.group(1).strip() if m else None
    for p in passages:
        if " — " in p:
            name, meaning = p.split(" — ", 1)
            if target and name.strip().lower() == target.lower():
                return meaning.strip()
    for p in passages:
        if " — " in p:
            return p.split(" — ", 1)[1].strip()
    return None

def build_prompt(context, question):
    return f"""Jawab dalam satu ayat pendek dalam Bahasa Melayu.
Jika terdapat maksud yang tepat dalam konteks, beri maksud itu sahaja tanpa huraian tambahan.

Konteks:
{context}

Soalan:
{question}

Jawapan:"""

def mk_q(name): return f"Apakah maksud peribahasa {name}?" if name else ""

# ---------- Lazy singletons ----------
_index = None
_pass   = None
_retr   = None
_tok    = None
_gen    = None
_last_cfg = None

def ensure_artifacts(use_lora: bool):
    global _index, _pass, _retr, _tok, _gen, _last_cfg
    # Index/passages/retriever
    if _index is None:
        _index = faiss.read_index(INDEX_PATH)
    if _pass is None:
        _pass = load_passages(PASSAGES)
    if _retr is None:
        # Allow hub id for retriever too
        retr_id = RETR_PATH if os.path.isdir(RETR_PATH) else RETR_PATH
        _retr = SentenceTransformer(retr_id)

    cfg = ("cpu", use_lora)
    if _gen is None or _last_cfg != cfg:
        _tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
        if _tok.pad_token is None: _tok.pad_token = _tok.eos_token
        _gen = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
        if use_lora:
            # If LORA_DIR is a directory in the repo, load from path. If you uploaded to Hub, use the repo id string instead.
            if os.path.isdir(LORA_DIR) or ("/" in LORA_DIR):
                try:
                    _gen = PeftModel.from_pretrained(_gen, LORA_DIR)
                except Exception:
                    pass
        _last_cfg = cfg

    return _index, _pass, _retr, _tok, _gen

def answer(question, top_k, use_extraction, use_lora):
    t0 = time.time()
    index, passages, retr, tok, gen = ensure_artifacts(use_lora=use_lora)

    # Retrieve
    q_emb = retr.encode([question], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    D, I = index.search(q_emb, int(top_k))
    topk_texts  = [passages[i] for i in I[0]]
    topk_scores = [float(s) for s in D[0]]

    # Exact extraction path
    if use_extraction:
        extracted = try_extract_exact_meaning(topk_texts, question)
        if extracted:
            return (
                question,
                "\n".join([f"{r+1}. ({topk_scores[r]:.4f})  {topk_texts[r]}" for r in range(len(topk_texts))]),
                extracted,
                f"{time.time()-t0:.2f}s",
                f"Model: {BASE_MODEL} | LoRA: {'ON' if use_lora else 'OFF'}"
            )

    # Generate (deterministic)
    ctx = "\n".join(topk_texts)
    ids = tok(build_prompt(ctx, question), return_tensors="pt")
    out = gen.generate(
        **ids, do_sample=False, temperature=1.0, top_p=1.0, top_k=0,
        max_new_tokens=32, eos_token_id=tok.eos_token_id
    )
    txt = tok.decode(out[0], skip_special_tokens=True)
    pred = txt.split("Jawapan:")[-1].strip()
    if "." in pred: pred = pred.split(".")[0].strip() + "."

    return (
        question,
        "\n".join([f"{r+1}. ({topk_scores[r]:.4f})  {topk_texts[r]}" for r in range(len(topk_texts))]),
        pred,
        f"{time.time()-t0:.2f}s",
        f"Model: {BASE_MODEL} | LoRA: {'ON' if use_lora else 'OFF'}"
    )

# ---------- Build UI ----------
from aksara_ui import build_ui

names = proverb_names_from_passages(PASSAGES, limit=300)
if BACKEND == "space":
    from remote_backend import answer_remote, SPACE_ID
    if os.environ.get("SPACE_ID") == SPACE_ID:
        raise ValueError("A Space cannot use itself as its remote backend")
    demo = build_ui(answer_remote, names, mk_q, BASE_MODEL, lambda: None, remote_space=SPACE_ID)
else:
    demo = build_ui(answer, names, mk_q, BASE_MODEL, lambda: isinstance(_gen, PeftModel))

if __name__ == "__main__":
    # Spaces auto-sets host/port; no need to pass server_name/port
    demo.queue(max_size=8).launch()
