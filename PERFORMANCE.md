# AKSARA progress and performance — 6 October 2026

## Current progress

The branded Gradio interface, example buttons, advanced controls, source cards,
technical log, error handling and remote Hugging Face API connection are implemented.
The local app runs in `space` mode, forwarding requests to
`Warisan2026/QA_Proverbs`. The configured generator is
`Qwen/Qwen2.5-1.5B-Instruct`. The remote API reports the requested LoRA setting;
it does not prove that the adapter actually loaded.

All 10 existing automated tests passed. They cover UI callbacks, output mapping,
escaping, error handling, concurrency, remote argument forwarding and lightweight
startup. They do not establish model accuracy. The test run emitted unclosed
asyncio event-loop ResourceWarnings, with no test failures.

## Live measurements

Measured sequentially through the local `/tanya_aksara` endpoint, including the
local Gradio queue, remote API request and network overhead. LoRA was requested
for every call. This is a small smoke test, not a load test or accuracy benchmark.

| Question | Top-k | Extraction | End-to-end | Backend | Observed answer |
| --- | ---: | --- | ---: | ---: | --- |
| makan puji, first call | 1 | On | 11.720 s | 0.38 s | Suka dipuji |
| makan puji, repeat | 1 | On | 4.988 s | 0.02 s | Suka dipuji |
| bagai aur dengan tebing | 1 | On | 3.583 s | 0.07 s | Duduk bawah budi orang |
| ribut dalam cawan | 1 | On | 3.205 s | 0.02 s | Awan mendung alamat akan ribut |
| buah tangan | 5 | On | 3.570 s | 0.02 s | Hasil perbuatan sendiri |
| makan puji | 1 | Off | 32.712 s | 30.37 s | Makan puji adalah memperolehi pujian atau kritik positif dari orang lain. |
| bagai aur dengan tebing, follow-up | 5 | On | 3.057 s | 0.02 s | Tolong menolong antara satu sama lain |

All seven requests completed. The first call includes remote client initialization;
it is not an isolated model cold-start measurement. Warm extraction calls took
3.057–4.988 seconds, with a median of 3.570 seconds.

The local HTML endpoint returned HTTP 200 in 32–89 ms over three requests; its
decoded body was 403,589 bytes. `/config` returned HTTP 200 in 44 ms. These numbers
measure local HTTP response time, not browser rendering, Core Web Vitals or public
website speed. No browser connection was available, so desktop/mobile visual QA
and browser performance remain unverified. Multi-user throughput was not measured.

## Issues found

1. **Extraction can silently answer a different proverb.** The implementation in
   `app.py:try_extract_exact_meaning` falls back to the first retrieved definition
   when it cannot find the requested name. Live results show this behavior too.
   Three of the four distinct example questions returned definitions attached to
   different proverbs under the tested settings. This is a release blocker for
   answer reliability, not an overall accuracy estimate.
2. **Top-k 1 misses an available exact entry.** For “bagai aur dengan tebing”, the
   exact entry ranked fourth. Top-k 5 recovered its corpus definition. Increasing
   Top-k alone did not resolve “buah tangan”.
3. **Corpus coverage affects the examples.** The local corpus contains 19,579
   passages and 19,281 distinct named entries. “ribut dalam cawan” and “buah tangan”
   have no exact named entries in that copy; “Oleh-oleh — Buah tangan” does appear.
4. **Generation needs quality evaluation.** With extraction disabled, “makan puji”
   produced wording different from its exact corpus definition “Suka dipuji”, and
   took approximately 33 seconds.
5. **The displayed latency is backend time.** It excludes several seconds of
   observed network/client/queue overhead. Inference is serialized with concurrency
   one and a queue limit of eight; multiple users may wait longer.

## Next work

Prioritize exact-name lookup before semantic retrieval and a clear no-match
response instead of returning an unrelated definition. Add answer-quality cases
covering all four visible examples, synonyms and missing corpus entries. Measure
user-visible elapsed time separately from backend time. Complete desktop/mobile
visual checks and a representative held-out model evaluation before release.

This check did not change the inference logic or deploy to Hugging Face. Changes
to local inference alone will not repair the active remote Space backend.
