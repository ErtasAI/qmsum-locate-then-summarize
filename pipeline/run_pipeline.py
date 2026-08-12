"""End-to-end: query -> locate -> budget-pack -> summarize -> predictions.

Per-query stage timings are recorded alongside every prediction: `locate_s` (the
cross-encoder scoring every window), `generate_s` (the summarizer forward pass) and
`latency_s` (their sum, which is what eval/run_eval.py averages into the row).

Two things make these numbers trustworthy rather than decorative:

1. `torch.cuda.synchronize()` brackets every timed region. CUDA kernels launch
   asynchronously, so timing them with a bare wall clock measures how fast Python
   queued the work, not how long the GPU took. Without the sync these numbers would
   be plausible and wrong.
2. The first query is flagged `warmup: true`. It carries CUDA context creation,
   kernel autotuning and lazy weight paging, and runs several times slower than
   steady state. It is written out rather than silently dropped so the exclusion is
   visible, and run_eval ignores it when averaging.

Latency is hardware- and load-dependent. Any reported figure must name the GPU and
state that the machine was otherwise idle, or it is not reproducible.
"""
import argparse, json, pathlib, time, torch
from data.normalize import load_meetings, load_queries
from pipeline.chunker import windows
from eval import protocol, vram

ROOT = pathlib.Path(__file__).resolve().parents[1]

def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def build_pipeline_prompt(query, utterances, locate_fn, span_budget=None, chunk_words=None):
    budget = span_budget or protocol.SPAN_BUDGET_WORDS
    picked = locate_fn(query, windows(utterances, chunk_words), budget)
    span_text = protocol.truncate_words("\n".join(p["text"] for p in picked), budget)
    return protocol.PROMPT_TEMPLATE.format(query=query, transcript=span_text)

def build_zeroshot_prompt(query, utterances, truncate_words):
    """Prompt for the no-locator zero-shot cells (declared 2026-08-12).

    The frozen template over the transcript truncated to a word budget: same content
    contract as build_pipeline_prompt, with the locate stage replaced by fixed-budget
    truncation. Used for base-model rows that measure what the pipeline looks like
    with no locator in front of the model.
    """
    transcript = protocol.truncate_words(protocol.format_transcript(utterances), truncate_words)
    return protocol.PROMPT_TEMPLATE.format(query=query, transcript=transcript)


def arg_error(adapter, base_model, locator, truncate_words, locator_ckpt, chunk_words,
              span_budget):
    """Reject invalid model/locator combinations before any weights load.

    Returns the error string or None. Pure so the contract is testable: exactly one
    model source, and the truncated-transcript mode cannot silently combine with
    locator-side settings whose provenance fields it would then misreport.
    """
    if (adapter is None) == (base_model is None):
        return "exactly one of --adapter / --base-model is required"
    if locator == "none":
        if not truncate_words:
            return "--locator none requires --truncate-words"
        if locator_ckpt or chunk_words or span_budget:
            return ("--locator none is incompatible with --locator-ckpt, --chunk-words "
                    "and --span-budget; the truncated-transcript mode has no locator side")
    elif truncate_words:
        return "--truncate-words requires --locator none"
    return None


def build_gen_kwargs(min_new_tokens=None, num_beams=None, length_penalty=None,
                     no_repeat_ngram_size=None, early_stopping=None):
    """Decode config, plus the subset of it that counts as a protocol deviation.

    Returns (kwargs, deviations). With no overrides, `kwargs` must be exactly the
    frozen greedy config and `deviations` must be empty, because the winner's 281
    test predictions were verified byte-identical on a re-run and that guarantee
    is only worth having if the default path cannot drift. Any override is a
    deviation and is written into every prediction record so a preds file can
    never be mistaken for a protocol run.

    `no_repeat_ngram_size` and `early_stopping` were added 2026-07-30 for the decode
    sweep. Both are in the published Socratic SegEnc config we now compare against
    (`num_beams: 4`, `no_repeat_ngram_size: 3`, `early_stopping: True`) while our own
    rows were generated greedily with no penalties, so the sweep is closing a
    self-imposed handicap rather than shopping for a decode that flatters us.
    """
    kwargs = {"max_new_tokens": protocol.MAX_NEW_TOKENS, "do_sample": False}
    for key, val in (("min_new_tokens", min_new_tokens), ("num_beams", num_beams),
                     ("length_penalty", length_penalty),
                     ("no_repeat_ngram_size", no_repeat_ngram_size),
                     ("early_stopping", early_stopping)):
        if val is not None:
            kwargs[key] = val
    deviations = {k: v for k, v in kwargs.items() if k not in ("max_new_tokens", "do_sample")}
    return kwargs, deviations


def main():
    from transformers import AutoTokenizer
    from peft import AutoPeftModelForCausalLM
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", help="LoRA adapter path or repo (the system rows)")
    ap.add_argument("--base-model",
                    help="run a base model zero-shot with NO adapter (the ablation cells). "
                         "The model gets its chat template; recorded in every prediction")
    ap.add_argument("--split", required=True, choices=["val", "test"])
    ap.add_argument("--locator", default="crossencoder",
                    choices=["embed", "crossencoder", "none"])
    ap.add_argument("--truncate-words", type=int,
                    help="with --locator none: feed the transcript truncated to this many "
                         "words instead of located spans (e.g. protocol.M1_TRUNCATE_WORDS)")
    ap.add_argument("--span-budget", type=int, default=None,
                    help="override protocol.SPAN_BUDGET_WORDS (sweep axis; a protocol "
                         "deviation, record it in the run notes)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int)
    # Decode overrides for the density probe. Every one of these is a PROTOCOL
    # DEVIATION and must be recorded as such in the run notes and any table.
    # They default to None so that omitting them reproduces the frozen greedy
    # config byte-identically: the kwargs below are only added when set.
    ap.add_argument("--min-new-tokens", type=int,
                    help="force at least this many new tokens (density probe)")
    ap.add_argument("--num-beams", type=int, help="beam search (deviation, default greedy)")
    ap.add_argument("--length-penalty", type=float, help="beam length penalty (deviation)")
    ap.add_argument("--no-repeat-ngram-size", type=int,
                    help="block repeated n-grams (deviation; Socratic SegEnc uses 3)")
    ap.add_argument("--early-stopping", action="store_true", default=None,
                    help="stop beams when all hypotheses finish (deviation; Socratic uses it)")
    # Locator overrides for the truncation experiment (the locator truncation analysis).
    # Both are protocol deviations and are recorded per prediction.
    ap.add_argument("--chunk-words", type=int,
                    help="window size in words (deviation; default protocol.CHUNK_WORDS=900). "
                         "MUST match the size the locator checkpoint was trained on.")
    ap.add_argument("--locator-ckpt",
                    help="path to a locator checkpoint other than the default "
                         "(deviation; e.g. checkpoints/locator-crossencoder-w375-l12)")
    a = ap.parse_args()
    err = arg_error(a.adapter, a.base_model, a.locator, a.truncate_words,
                    a.locator_ckpt, a.chunk_words, a.span_budget)
    if err:
        raise SystemExit(err)
    locate = None
    if a.locator == "embed":
        from pipeline.locator_embed import select as locate
        if a.locator_ckpt:
            raise SystemExit("--locator-ckpt applies to the crossencoder locator only")
    elif a.locator == "crossencoder":
        from pipeline import locator_crossencoder as _lc
        if a.locator_ckpt:
            # Rebind before the lazy loader runs, so the override cannot be missed.
            _lc.CKPT, _lc._MODEL = pathlib.Path(a.locator_ckpt), None
        from pipeline.locator_crossencoder import select as locate
    if a.base_model:
        from transformers import AutoModelForCausalLM
        tok = AutoTokenizer.from_pretrained(a.base_model)
        model = AutoModelForCausalLM.from_pretrained(a.base_model, device_map="cuda",
                                                     torch_dtype=torch.bfloat16)
    else:
        tok = AutoTokenizer.from_pretrained(a.adapter)
        model = AutoPeftModelForCausalLM.from_pretrained(a.adapter, device_map="cuda",
                                                         torch_dtype=torch.bfloat16)
    model.eval()
    gen_kwargs, deviations = build_gen_kwargs(a.min_new_tokens, a.num_beams,
                                              a.length_penalty, a.no_repeat_ngram_size,
                                              a.early_stopping)
    # Beam search on an LFM2 hybrid cache is broken in transformers 4.57.6: it raises on the
    # conv layers' empty key/value placeholders, and silently fails to reorder the trailing
    # conv layer's state. Patch ONLY on the beam path so the frozen greedy config, whose 281
    # test predictions were verified byte-identical on a re-run, cannot be perturbed.
    if (a.num_beams or 1) > 1:
        from pipeline import lfm2_beam_fix
        lfm2_beam_fix.apply()
        deviations["lfm2_beam_cache_fix"] = True
    if a.chunk_words:
        deviations["chunk_words"] = a.chunk_words
    if a.locator_ckpt:
        deviations["locator_ckpt"] = a.locator_ckpt
    if a.span_budget:
        deviations["span_budget_words"] = a.span_budget
    meetings = {m["meeting_id"]: m["utterances"] for m in load_meetings(a.split)}
    out = pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    with out.open("w", encoding="utf-8") as f, torch.no_grad():
        for i, q in enumerate(load_queries(a.split)[:a.limit]):
            # The peak window opens before LOCATE, so the recorded figure is what the whole
            # system needs, both models resident, not what the summarizer needs alone. The
            # locate-only peak is read out separately below without a second reset, so the
            # split is visible and the total stays a true maximum over the query.
            vram.reset()
            _sync(); t0 = time.perf_counter()
            if a.locator == "none":
                prompt = build_zeroshot_prompt(q["query"], meetings[q["meeting_id"]],
                                               a.truncate_words)
            else:
                prompt = build_pipeline_prompt(q["query"], meetings[q["meeting_id"]], locate,
                                               a.span_budget, a.chunk_words)
            _sync(); t1 = time.perf_counter()
            peak_locate = vram.peak_gb()

            if a.base_model:
                # Zero-shot base rows get the model's own chat template with the frozen
                # template's text as the single user message, mirroring the frontier
                # baselines' chat-API treatment (declared 2026-08-12).
                ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                              add_generation_prompt=True,
                                              return_tensors="pt",
                                              return_dict=True).to("cuda")
            else:
                ids = tok(prompt + " ", return_tensors="pt").to("cuda")
            gen = model.generate(**ids, **gen_kwargs)
            _sync(); t2 = time.perf_counter()

            n_in = ids["input_ids"].shape[1]
            n_new = gen.shape[1] - n_in
            text = tok.decode(gen[0][n_in:], skip_special_tokens=True)
            f.write(json.dumps({
                "query_id": q["query_id"], "prediction": text.strip(),
                "locate_s": round(t1 - t0, 4),
                "generate_s": round(t2 - t1, 4),
                "latency_s": round(t2 - t0, 4),
                "prompt_tokens": int(n_in), "new_tokens": int(n_new),
                "warmup": i == 0, "device": gpu,
                **vram.record(),
                **({"peak_vram_locate_gb": peak_locate} if peak_locate is not None else {}),
                **({"base_model": a.base_model, "prompt_format": "chat"}
                   if a.base_model else {}),
                **({"truncate_words": a.truncate_words} if a.locator == "none" else {}),
                **({"decode_deviations": deviations} if deviations else {})}) + "\n")
            del ids, gen
            torch.cuda.empty_cache()
    print(f"wrote {out}")

if __name__ == "__main__":
    main()
