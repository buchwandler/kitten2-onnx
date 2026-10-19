# Architecture

## Upstream KittenTTS 2 pipeline

KittenTTS 2 is not the older single-graph Kitten ONNX architecture. Its core
pipeline is:

```text
text + reference voice
        |
        v
generation prompt
        |
        +---------------- speaker embedding [512]
        |                         |
        |                         v
        |                      spk_proj
        |                         |
        v                         v
token IDs -> embeddings -> replace embedding position 0
                         |
                         v
                Qwen3ForCausalLM
                         |
                         v
                  S3 codec tokens
                         |
                         v
                     S3Gen
                         |
                         v
                  24 kHz waveform
```

The upstream implementation uses a standard Hugging Face
`AutoModelForCausalLM`/Qwen3 architecture. `spk_proj` is checkpoint-specific and
is attached separately because the base Qwen3 model does not define it.

## MVP graph split

The first version intentionally uses small, composable contracts:

```text
embed_tokens.npy
      |
token ids -> lookup
      |
speaker[512] -> speaker.onnx -> hidden vector
      |                    |
      +------ replace position 0
                           |
                           v
                       lm.onnx
                  inputs_embeds + KV
                           |
                           v
                         logits
```

`lm.onnx` returns last-position logits and full present KV tensors. A single graph
serves both prefill and decode.

## Why not a monolithic graph?

Keeping orchestration outside ONNX makes the first parity milestone much easier:

- Kitten has custom logits processors and sampling policy.
- speaker injection happens only at prefill.
- the generation loop naturally owns the KV cache.
- the S3 audio decoder is a separate subsystem.
- failures can be localized to LM, speaker projection, sampling, or codec.

A later runtime can fuse or specialize parts after parity is established.

## Codec boundary

Upstream S3Gen performs both reference conditioning and generated-token decoding.
The initial plan is to validate three boundaries independently:

1. reference conditioning;
2. S3 token + conditioning -> mel/flow output;
3. mel -> 24 kHz waveform.

Do not claim an end-to-end ONNX KittenTTS 2 runtime until the waveform path has
objective and listening parity tests.


## Text preprocessing and emotion conditioning

The PyTorch reference and ONNX runtime share `prepare_generation_text` before prompt construction:

```text
original text
    |
    +--> protect expression and vocal-event markup
    |
    +--> optional language normalization (`normalize=True` by default)
    |
    +--> restore markup and construct prompt IDs
    |
    +--> resolve configured emotion-control token
```

Expression and vocal-event tags are protected from rewriting. `use_emotion=None` enables the configured control when recognized markup is present. Explicit `True` and `False` force or suppress the control. The result records the detected markup and whether emotion conditioning was used.

`normalize=False` skips the language normalizer but retains basic formatting and sentence termination. Unicode terminal punctuation for Arabic, Chinese, and Devanagari is treated as terminal punctuation, rather than receiving an extra ASCII period. The release corpus uses this mode for German, Chinese, Arabic, and Hindi and compares prompt IDs, logits, generated token IDs, and finite audio between PyTorch and ONNX.

## Voice packages and local bundle augmentation

A built-in voice is reference conditioning data, not a separate set of ONNX graphs. `bundle.json` maps each voice name to a decoder-reference package. The speaker embedding, reference tokens, conditioning tensors, transcript metadata, and captured Flow inputs vary by voice. The speaker, LM, Flow, and mel-to-waveform graphs remain shared. Bruno remains the default voice.

The accepted 47-name voice set is pinned by `voices/voices.json` at model revision `fa8ce5c7547e322a065cad9b813d27a78fad9640` and upstream code revision `36ef3090ca1e59122a81fc23be4a659a502b9d55`. The all-voice capture runner reuses one S3Gen context, processes names sequentially, validates each `.npz` and metadata pair, writes each pair atomically, and skips valid captures on resume. The S3Gen checkpoint digest is computed once per run and reused across voices.

For an existing ONNX bundle, `scripts/augment_bundle_voices.py` performs a local promotion transaction:

1. Validate and hash the source bundle.
2. Copy the source directory to a sibling staging directory.
3. Add missing voice references and regenerate the staged `bundle.json` file hashes.
4. Validate all staged files, dynamic axes, static tensor dimensions, voice metadata, token ranges, and finite values.
5. Compare all unchanged source file hashes, excluding the intentionally regenerated `bundle.json`.
6. Promote only the validated staging directory with an atomic rename.

The output must be a new path. The source bundle and its non-manifest files remain unchanged. The tool does not upload or write to the Hugging Face repository. Direct export from the reference model is a separate path; `export_all --all-voices` captures all selected voices while retaining one shared decoder graph pair.

## Release evidence and resource boundaries

`build/reports/feature-parity-47voices.json` summarizes the local release evidence. At the recorded pinned revisions, the augmented bundle contained 47 voices and passed validation for all 167 listed files. The augmentation report recorded preserved hashes for every source file other than the regenerated manifest. Exact parity passed for three expression cases and four multilingual cases, with finite waveforms. A non-default Sable voice also passed the synthesis CLI smoke check.

The same evidence records measured resource use: all-voice capture peaked near 3.03 GB RSS; augmentation peaked near 3.69 GB; expression parity peaked near 15.88 GB; multilingual parity peaked near 16.04 GB. Capture and release validation are sequential. Full CPU parity needs substantially more memory than packaging, so run it through `scripts/run_bounded.py`, check host availability, set an explicit RSS cap and available-memory reserve, and preserve guard results. The README records elapsed times and the settings used for this run. These are host measurements, not universal sizing guarantees.

The release sequence is to pin model and upstream revisions, download and capture locally, augment into a new path, validate hashes and structure, run expression and multilingual parity plus finite-audio checks, then review the resulting reports. Remote publication is outside this local build boundary.
