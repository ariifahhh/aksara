import os, json, time, faiss, re
import gradio as gr
import numpy as np

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ---------------- CONFIG ----------------
DEFAULT_FORCE_CPU = True
INDEX_PATH    = r"C:\Users\Admin\Desktop\provmalay\faiss.index"
PASSAGES_PATH = r"C:\Users\Admin\Desktop\provmalay\passages.jsonl"
RETR_MODEL    = r"C:\Users\Admin\Desktop\provmalay\retriever_model"
BASE_MODEL    = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_DIR      = r"C:\Users\Admin\Desktop\provmalay\qwen-peribahasa-lora"

# ---------------- UTILITIES ----------------
def load_passages(pfile):
    texts = []
    with open(pfile, "r", encoding="utf-8") as f:
        for line in f:
            texts.append(json.loads(line)["text"])
    return texts

def get_proverb_names(limit=300):
    """Parse '<peribahasa> — <maksud>' lines to collect unique proverb names for the dropdown."""
    names, seen = [], set()
    for line in open(PASSAGES_PATH, encoding="utf-8"):
        obj = json.loads(line)
        t = obj["text"]
        if " — " in t:
            name = t.split(" — ", 1)[0].strip()
            if name and name.lower() not in seen:
                seen.add(name.lower())
                names.append(name)
                if len(names) >= limit:
                    break
    # Keep a few curated, common ones at the top if present
    pin = ["Makan puji", "Makan puluk", "Makan pokok", "Makan ransom", "Pujaan"]
    names = [n for n in pin if n in names] + [n for n in names if n not in pin]
    return names

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

def make_q_from_name(name: str):
    return f"Apakah maksud peribahasa {name}?" if name else ""

# ---------------- LAZY LOAD GLOBALS ----------------
_index = None
_passages = None
_retr = None
_tok = None
_gen = None
_last_cfg = {"force_cpu": None, "use_lora": None, "base": None, "lora_dir": None}

def ensure_index_and_retr():
    global _index, _passages, _retr
    if _index is None:
        _index = faiss.read_index(INDEX_PATH)
    if _passages is None:
        _passages = load_passages(PASSAGES_PATH)
    if _retr is None:
        _retr = SentenceTransformer(RETR_MODEL)
    return _index, _passages, _retr

def ensure_generator(force_cpu: bool, use_lora: bool, base: str, lora_dir: str):
    global _tok, _gen, _last_cfg
    cfg = {"force_cpu": force_cpu, "use_lora": use_lora, "base": base, "lora_dir": lora_dir}
    if _gen is not None and cfg == _last_cfg:
        return _tok, _gen

    device = "cpu" if force_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    if force_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    _tok = AutoTokenizer.from_pretrained(base, use_fast=True)
    if _tok.pad_token is None:
        _tok.pad_token = _tok.eos_token

    if device == "cpu":
        _gen = AutoModelForCausalLM.from_pretrained(base)
    else:
        _gen = AutoModelForCausalLM.from_pretrained(base, torch_dtype="auto", device_map="auto")

    if use_lora and lora_dir and os.path.isdir(lora_dir):
        try:
            _gen = PeftModel.from_pretrained(_gen, lora_dir)
        except Exception:
            pass

    _last_cfg = cfg
    return _tok, _gen

# ---------------- INFERENCE ----------------
def answer(question, top_k, use_extraction, use_lora, force_cpu, base_model_id, lora_dir):
    t0 = time.time()
    index, passages, retr = ensure_index_and_retr()
    tok, gen = ensure_generator(force_cpu=force_cpu, use_lora=use_lora, base=base_model_id, lora_dir=lora_dir)

    q_emb = retr.encode([question], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    D, I = index.search(q_emb, int(top_k))
    topk_texts = [passages[idx] for idx in I[0]]
    topk_scores = [float(s) for s in D[0]]

    if use_extraction:
        extracted = try_extract_exact_meaning(topk_texts, question)
        if extracted:
            return (
                question,
                "\n".join([f"{i+1}. ({topk_scores[i]:.4f})  {topk_texts[i]}" for i in range(len(topk_texts))]),
                extracted,
                f"{time.time()-t0:.2f}s"
            )

    ctx = "\n".join(topk_texts)
    prompt = build_prompt(ctx, question)
    ids = tok(prompt, return_tensors="pt")
    out = gen.generate(
        **ids, do_sample=False, temperature=1.0, top_p=1.0, top_k=0,
        max_new_tokens=32, eos_token_id=tok.eos_token_id
    )
    text = tok.decode(out[0], skip_special_tokens=True)
    pred = text.split("Jawapan:")[-1].strip()
    if "." in pred:
        pred = pred.split(".")[0].strip() + "."

    return (
        question,
        "\n".join([f"{i+1}. ({topk_scores[i]:.4f})  {topk_texts[i]}" for i in range(len(topk_texts))]),
        pred,
        f"{time.time()-t0:.2f}s"
    )

# ---------------- UI ----------------
# Build examples from a few common names
_proverb_names = get_proverb_names(limit=300)
_example_qs = [
    [make_q_from_name(n)]
    for n in (_proverb_names[:5] if len(_proverb_names) >= 5 else ["Makan puji","Makan puluk","Makan pokok","Makan ransom","Pujaan"])
]

with gr.Blocks(title="Peribahasa RAG — Qwen + LoRA") as demo:
    gr.Markdown(
        "## 🇲🇾 My Peribahasa QA — Retrieval-Augmented Generation (Qwen + LoRA)\n"
        "Type or select a proverb. The app retrieves relevant lines and answers in a single short sentence."
    )

    with gr.Row():
        with gr.Column(scale=2):
            q_in = gr.Textbox(label="Soalan (Malay)", value="Apakah maksud peribahasa Makan puji?")
            # NEW: Examples (click to fill the question box)
            gr.Examples(
                label="Contoh soalan (klik untuk guna)",
                examples=_example_qs,
                inputs=q_in
            )
            # NEW: Dropdown of peribahasa (auto from passages.jsonl)
            with gr.Row():
                dd = gr.Dropdown(
                    choices=_proverb_names,
                    label="Pilih peribahasa (dropdown)",
                    value=_proverb_names[0] if _proverb_names else None
                )
                fill_btn = gr.Button("Gunakan peribahasa terpilih", size="sm")
            topk = gr.Slider(1, 5, value=1, step=1, label="Top-k (retrieval)")
            use_extraction = gr.Checkbox(value=True, label="Use exact-gloss extraction (recommended)")
        with gr.Column(scale=1):
            use_lora = gr.Checkbox(value=True, label="Use LoRA adapter (if available)")
            force_cpu = gr.Checkbox(value=DEFAULT_FORCE_CPU, label="Force CPU (safer while training)")
            base_model_id = gr.Textbox(label="Base model ID", value=BASE_MODEL)
            lora_dir_in = gr.Textbox(label="LoRA directory", value=LORA_DIR)

    go = gr.Button("Jawab", variant="primary")
    with gr.Row():
        q_out = gr.Textbox(label="Question (echo)", interactive=False)
    with gr.Row():
        ctx_out = gr.Textbox(label="Retrieved Context", lines=6, interactive=False)
    with gr.Row():
        ans_out = gr.Textbox(label="Answer", lines=2, interactive=False)
    with gr.Row():
        t_out = gr.Textbox(label="Latency", interactive=False)

    # Hook: dropdown → question box
    fill_btn.click(lambda name: make_q_from_name(name), inputs=dd, outputs=q_in)

    go.click(
        fn=answer,
        inputs=[q_in, topk, use_extraction, use_lora, force_cpu, base_model_id, lora_dir_in],
        outputs=[q_out, ctx_out, ans_out, t_out],
        api_name="answer"
    )

if __name__ == "__main__":
    demo.queue(max_size=8).launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
