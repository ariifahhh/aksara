"""Use the existing Space's Gradio endpoint without loading local model weights."""

import os
from functools import lru_cache

from gradio_client import Client

SPACE_ID = "Warisan2026/QA_Proverbs"


@lru_cache(maxsize=1)
def get_client():
    return Client(SPACE_ID, hf_token=os.environ.get("HF_TOKEN") or None, verbose=False)


def answer_remote(question, top_k, use_extraction, use_lora):
    result = get_client().predict(
        question, top_k, use_extraction, use_lora, api_name="/answer"
    )
    if not isinstance(result, (tuple, list)) or len(result) != 5:
        raise RuntimeError("The Space returned an unexpected answer format")
    return tuple(result)
