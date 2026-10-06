"""AKSARA presentation layer. Model loading and inference stay in app.py."""

import base64
import html
import logging
import re
from pathlib import Path

import gradio as gr

ASSETS = Path(__file__).resolve().parent / "assets"
LOG = logging.getLogger(__name__)
EXAMPLES = [
    "Apakah maksud peribahasa makan puji?",
    "Apakah maksud peribahasa bagai aur dengan tebing?",
    "Apakah maksud peribahasa ribut dalam cawan?",
    "Apakah maksud peribahasa buah tangan?",
]


def logo():
    data = base64.b64encode((ASSETS / "aksara-icon.png").read_bytes()).decode("ascii")
    return f'<img class="aksara-logo" src="data:image/png;base64,{data}" alt="Logo AKSARA" width="52" height="52">'


def icon(kind):
    paths = {
        "book": '<path d="M12 5v15M3 4c4-1 6 0 9 2 3-2 5-3 9-2v14c-4-1-6 0-9 2-3-2-5-3-9-2Z"/>',
        "model": '<rect x="5" y="5" width="14" height="14" rx="3"/><path d="M9 1v4m6-4v4M9 19v4m6-4v4M1 9h4m-4 6h4m14-6h4m-4 6h4"/><rect x="9" y="9" width="6" height="6" rx="1"/>',
        "search": '<circle cx="10" cy="10" r="6"/><path d="m15 15 6 6M7 10h6m-3-3v6"/>',
    }
    return f'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4" aria-hidden="true">{paths[kind]}</svg>'


def render_context(raw=""):
    """Decorate the original retrieval text without reranking or dropping content."""
    if not raw:
        return '<div class="empty-state"><span class="empty-symbol" aria-hidden="true">◇</span><h3>Sumber untuk setiap makna</h3><p>Konteks daripada korpus akan dipaparkan selepas anda menghantar soalan.</p></div>'
    # Entries are emitted by answer() as `1. (0.1234)  text`; preserve multiline text.
    entries = re.split(r"(?m)(?=^\d+\. \([-+\d.eE]+\)  )", raw)
    cards = []
    for entry in filter(None, entries):
        match = re.match(r"(\d+)\. \(([-+\d.eE]+)\)  ([\s\S]*)", entry)
        if match:
            number, score, text = match.groups()
            title, separator, body = text.partition(" — ")
            content = (
                f"<h3>{html.escape(title)}</h3><p>{html.escape(body)}</p>"
                if separator
                else f"<p>{html.escape(text)}</p>"
            )
            cards.append(
                f'<article class="context-card"><div class="context-meta"><span>Sumber {number}</span><span>Skor {html.escape(score)}</span></div>{content}</article>'
            )
        else:
            cards.append(
                f'<article class="context-card"><p>{html.escape(entry)}</p></article>'
            )
    return '<div class="context-list">' + "".join(cards) + "</div>"


def render_status(label="Sedia menerima soalan", latency=None, top_k=None, lora=None):
    details = ""
    if latency is not None:
        adapter = (
            "Tidak disahkan oleh API"
            if lora is None
            else ("Aktif" if lora else "Tidak aktif")
        )
        details = f"<dl><div><dt>Masa respons</dt><dd>{html.escape(latency)}</dd></div><div><dt>Top-k</dt><dd>{int(top_k)}</dd></div><div><dt>Adapter LoRA</dt><dd>{adapter}</dd></div></dl>"
    return f'<div class="status-card" role="status" aria-live="polite"><div class="status-label"><span aria-hidden="true">◈</span> {html.escape(label)}</div>{details}</div>'


def build_ui(
    answer_fn, names, make_question, base_model, lora_loaded, remote_space=None
):
    """Bind the unchanged five-value backend result to branded Gradio components."""
    brand = logo()
    backend_note = (
        f"**Pemprosesan:** API Hugging Face `{remote_space}`. Soalan dan tetapan dihantar ke Space ini. Masa respons di atas ialah masa backend; masa rangkaian dan giliran mungkin menambah masa menunggu. API melaporkan pilihan LoRA, bukan pengesahan adapter dimuatkan."
        if remote_space
        else "Status backend mengekalkan laporan asal. Medan LoRA di situ menunjukkan pilihan pengguna; kad status memaparkan sama ada adapter benar-benar dimuatkan."
    )
    theme = gr.themes.Base(
        primary_hue="teal",
        secondary_hue="cyan",
        neutral_hue="slate",
        font=["Inter", "Arial", "sans-serif"],
    ).set(
        body_background_fill="#061723",
        body_background_fill_dark="#061723",
        body_text_color="#F5F7FA",
        body_text_color_dark="#F5F7FA",
        block_background_fill="#071f2d",
        block_background_fill_dark="#071f2d",
        block_border_color="#27434d",
        block_border_color_dark="#27434d",
        block_label_text_color="#B9C5CF",
        block_label_text_color_dark="#B9C5CF",
        input_background_fill="#061b29",
        input_background_fill_dark="#061b29",
        input_border_color="#315361",
        input_border_color_dark="#315361",
        input_placeholder_color="#91a8b8",
        input_placeholder_color_dark="#91a8b8",
        button_secondary_background_fill="#092b3b",
        button_secondary_background_fill_dark="#092b3b",
        button_secondary_text_color="#F5F7FA",
        button_secondary_text_color_dark="#F5F7FA",
        border_color_primary="#315361",
        border_color_primary_dark="#315361",
        background_fill_primary="#071f2d",
        background_fill_primary_dark="#071f2d",
        background_fill_secondary="#0a2938",
        background_fill_secondary_dark="#0a2938",
    )
    with gr.Blocks(
        theme=theme,
        css=(ASSETS / "aksara.css").read_text(encoding="utf-8"),
        title="AKSARA · AI Bahasa & Warisan Melayu",
    ) as demo:
        gr.HTML(f"""
        <header class="aksara-navbar" id="utama">
          <a class="brand" href="#utama">{brand}<span><strong>AKSARA</strong><small>AI Bahasa &amp; Warisan Melayu</small></span></a>
          <nav aria-label="Navigasi utama"><a class="nav-active" href="#utama">Utama</a><a href="#tentang">Tentang</a><a href="#contoh-soalan">Contoh Soalan</a><a href="#dokumentasi">Dokumentasi</a></nav>
          <a class="nav-cta" href="#soalan">Cuba Sekarang <span aria-hidden="true">↗</span></a>
        </header>
        <section class="hero-section" aria-labelledby="hero-title">
          <div class="hero-copy"><p class="eyebrow"><span></span> BAHASA JIWA BANGSA</p>
          <h1 id="hero-title">Temui Makna Peribahasa dan<br><em>Warisan Melayu</em> Melalui Bahasa</h1>
          <p class="hero-description">Terokai khazanah peribahasa Melayu bersama pembantu AI,<br class="desktop-break"> dengan jawapan berpandukan korpus warisan kita.</p>
          <p class="hero-tagline">Meneroka Warisan Melalui Bahasa</p></div>
          <div class="hero-ornament" aria-hidden="true"><div class="ornament-ring"></div><span>❖</span><small>ILMU · BAHASA · WARISAN</small></div>
        </section>
        <div class="heritage-divider" aria-hidden="true"><span>◇</span></div>
        """)
        with gr.Row(elem_id="workspace", equal_height=False):
            with gr.Column(scale=4, min_width=320, elem_classes=["main-column"]):
                with gr.Group(elem_classes=["main-app-card"]):
                    gr.HTML(
                        '<div class="section-heading"><div><p class="eyebrow">RUANG PENEROKAAN</p><h2>Tanya AKSARA</h2></div><span class="section-badge">Peribahasa Melayu</span></div>'
                    )
                    gr.HTML(
                        '<h3 class="field-heading" id="contoh-soalan">Contoh Soalan <span>Mulakan dengan satu pertanyaan</span></h3>'
                    )
                    example_buttons = []
                    for offset in (0, 2):
                        with gr.Row(elem_classes=["example-row"]):
                            for question in EXAMPLES[offset : offset + 2]:
                                example_buttons.append(
                                    gr.Button(
                                        question + "  ↗",
                                        size="sm",
                                        elem_classes=["question-card"],
                                    )
                                )
                    q_in = gr.Textbox(
                        label="Soalan",
                        value="Apakah maksud peribahasa Makan puji?",
                        placeholder="Tanya apa-apa tentang peribahasa Melayu...",
                        lines=2,
                        max_lines=6,
                        elem_id="soalan",
                        elem_classes=["aksara-input"],
                    )
                    with gr.Row(elem_classes=["submit-row"]):
                        gr.HTML(
                            '<p class="input-hint">Setiap pertanyaan membuka lembaran baharu.</p>'
                        )
                        go = gr.Button(
                            "Tanya AKSARA  ↗",
                            variant="primary",
                            elem_classes=["aksara-primary-btn"],
                            scale=0,
                            min_width=190,
                        )
                    with gr.Accordion(
                        "Tetapan Lanjutan", open=False, elem_classes=["settings-card"]
                    ):
                        with gr.Row():
                            dd = gr.Dropdown(
                                choices=names,
                                label="Pilih Peribahasa",
                                value=names[0] if names else None,
                                scale=3,
                            )
                            fill_btn = gr.Button("Gunakan peribahasa terpilih", scale=1)
                        topk = gr.Slider(1, 5, value=1, step=1, label="Top-k Retrieval")
                        with gr.Row():
                            use_extraction = gr.Checkbox(
                                value=True, label="Pengekstrakan glos tepat (disyorkan)"
                            )
                            use_lora = gr.Checkbox(
                                value=True, label="Gunakan adapter LoRA jika tersedia"
                            )
                        gr.Markdown(
                            "Top-k menentukan bilangan petikan korpus. Glos tepat menggunakan maksud daripada konteks apabila ditemui; pilihan LoRA digunakan untuk penjanaan model.",
                            elem_classes=["settings-help"],
                        )
                with (
                    gr.Group(elem_classes=["results-panel"]),
                    gr.Tabs(elem_id="result-tabs"),
                ):
                    with gr.Tab("Jawapan"):
                        gr.HTML(
                            '<div class="answer-heading"><span aria-hidden="true">✧</span><h2>Makna di sebalik kata</h2></div>'
                        )
                        ans_out = gr.Textbox(
                            label="Jawapan AKSARA",
                            placeholder="Jawapan anda akan muncul di sini.",
                            lines=3,
                            interactive=False,
                            show_copy_button=True,
                            elem_classes=["answer-card"],
                        )
                        gr.HTML(
                            '<p class="answer-note">Berpandukan korpus peribahasa Melayu. Semak petikan sumber dalam tab Konteks Diperoleh.</p>'
                        )
                    with gr.Tab("Konteks Diperoleh"):
                        context_cards = gr.HTML(
                            render_context(), elem_id="context-cards"
                        )
                        with gr.Accordion("Lihat konteks asal", open=False):
                            ctx_out = gr.Textbox(
                                label="Retrieved Context",
                                lines=6,
                                interactive=False,
                                show_copy_button=True,
                            )
                    with gr.Tab("Log Teknikal"):
                        q_out = gr.Textbox(label="Soalan dihantar", interactive=False)
                        with gr.Row():
                            lat_out = gr.Textbox(
                                label="Masa respons", interactive=False
                            )
                            stat_out = gr.Textbox(
                                label="Status backend", interactive=False, scale=3
                            )
                        gr.Markdown(
                            f"**Model:** `{base_model}`\n\n**Kaedah:** Sentence Transformer → FAISS → glos tepat atau penjanaan Qwen.\n\n{backend_note}"
                        )
            with gr.Column(scale=1, min_width=265, elem_classes=["sidebar-column"]):
                gr.HTML(
                    f"""<aside class="sidebar-card" id="tentang">{brand}<p class="eyebrow">KENALI AKSARA</p><h2>Bahasa menghubungkan<br><em>kita dengan warisan.</em></h2><p>AKSARA membantu anda meneroka makna peribahasa dan simpulan bahasa Melayu melalui pencarian korpus dan jawapan AI.</p><div class="sidebar-signature">Khazanah bahasa, di hujung jari.</div></aside>"""
                )
                for kind, title, description in [
                    (
                        "book",
                        "Korpus Warisan Melayu",
                        "Koleksi peribahasa dan maksudnya sebagai rujukan jawapan.",
                    ),
                    (
                        "model",
                        "Qwen + LoRA",
                        "Model Qwen dengan adapter pilihan untuk penjanaan jawapan Bahasa Melayu.",
                    ),
                    (
                        "search",
                        "Berpandukan sumber",
                        "Retrieval-Augmented Generation (RAG) menghubungkan soalan dengan petikan korpus.",
                    ),
                ]:
                    gr.HTML(
                        f'<div class="feature-card"><span class="feature-icon">{icon(kind)}</span><div><h3>{title}</h3><p>{description}</p></div></div>'
                    )
                status_card = gr.HTML(render_status(), elem_id="runtime-status")
                if remote_space:
                    gr.Markdown(
                        "Jawapan diproses melalui **Hugging Face Space**. Tiada model dimuat turun pada komputer ini.",
                        elem_classes=["settings-help"],
                    )
        with gr.Accordion("Panduan & Dokumentasi", open=False, elem_id="dokumentasi"):
            gr.Markdown("""### Meneroka bersama AKSARA
1. Taip soalan atau pilih salah satu contoh, kemudian tekan **Tanya AKSARA**.
2. Buka **Tetapan Lanjutan** untuk memilih peribahasa, Top-k (1–5), glos tepat atau adapter LoRA.
3. Baca **Jawapan**, semak **Konteks Diperoleh**, atau lihat **Log Teknikal** untuk laporan backend.

Respons pertama mungkin mengambil masa lebih lama jika backend belum sedia. Jawapan boleh tersilap; gunakan petikan korpus sebagai rujukan.
""")
        gr.HTML(
            f"""<footer class="aksara-footer"><a class="brand" href="#utama">{brand}<span><strong>AKSARA</strong><small>AI Bahasa &amp; Warisan Melayu</small></span></a><nav aria-label="Navigasi bawah"><a href="#utama">Utama</a><a href="#tentang">Tentang</a><a href="#contoh-soalan">Contoh Soalan</a><a href="#dokumentasi">Dokumentasi</a></nav><span>Dibangunkan untuk <b>WaRiSAN Melayu</b></span></footer>"""
        )

        for button, question in zip(example_buttons, EXAMPLES):
            button.click(
                lambda q=question: q, outputs=q_in, queue=False, api_name=False
            )
        fill_btn.click(
            make_question,
            inputs=dd,
            outputs=q_in,
            queue=False,
            api_name="pilih_peribahasa",
        )

        def present_answer(question, top_k, extraction, lora):
            if not question or not question.strip():
                raise gr.Error("Sila masukkan soalan terlebih dahulu.")
            yield (
                "",
                "",
                "",
                "",
                "",
                render_context(""),
                render_status("Sedang mencari dan menyediakan jawapan…"),
            )
            try:
                result = answer_fn(question, top_k, extraction, lora)
                yield (
                    *result,
                    render_context(result[1]),
                    render_status("Jawapan tersedia", result[3], top_k, lora_loaded()),
                )
            except Exception:
                LOG.exception("AKSARA could not answer the question")
                yield (
                    "",
                    "",
                    "",
                    "",
                    "",
                    render_context(""),
                    render_status("Jawapan tidak dapat dijana"),
                )
                message = (
                    "Maaf, API Hugging Face tidak dapat memberikan jawapan. Semak sambungan internet dan status Space, kemudian cuba semula."
                    if remote_space
                    else "Maaf, jawapan tidak dapat dijana. Sila cuba semula."
                )
                raise gr.Error(message) from None

        inputs = [q_in, topk, use_extraction, use_lora]
        outputs = [
            q_out,
            ctx_out,
            ans_out,
            lat_out,
            stat_out,
            context_cards,
            status_card,
        ]
        go.click(
            present_answer,
            inputs=inputs,
            outputs=outputs,
            concurrency_id="aksara-model",
            concurrency_limit=1,
            api_name="tanya_aksara",
        )
        q_in.submit(
            present_answer,
            inputs=inputs,
            outputs=outputs,
            concurrency_id="aksara-model",
            concurrency_limit=1,
            api_name=False,
        )
    return demo
