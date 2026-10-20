# Shared LM vocabulary runtime

## Graph contract

The opt-in `export_lm --shared-vocabulary` mode writes a vocabulary-free transformer graph to `lm.onnx` and a separate `lm_output.onnx` projection graph. The transformer graph accepts `inputs_embeds` and returns `last_hidden_state` plus the KV cache. The projection graph accepts `hidden_states` and `vocabulary_weight` as inputs and returns logits. It contains no vocabulary initializer or external vocabulary data.

The runtime loads one `SharedVocabulary` object from the embedding asset. That object reconstructs the selected table as float32, uses it to look up token rows, and creates one CPU `OrtValue` used by the output head through ONNX Runtime I/O binding. No second vocabulary matrix is stored in either graph.

## Initializer override probe and fallback

The CPU initializer override probe uses `SessionOptions.add_initializer` with an ONNX `MatMul` initializer. Under the tested ONNX Runtime 1.31.0 Linux CPU environment, session creation terminated the isolated probe process with SIGSEGV. The probe runs in a subprocess so this behavior cannot terminate the test runner.

Because initializer override is not safe in that tested runtime, shared output projection uses the graph-input/I/O-binding path. The runtime binds the shared vocabulary `OrtValue` once per output-head run and binds only the small hidden-state input separately. `tests/test_onnx_export_tiny.py` exercises prefill and decode on CPU and compares the split runtime against the existing tied-head wrapper. `tests/test_runtime_shared_vocabulary.py` isolates the initializer override probe.

This records behavior observed with ONNX Runtime 1.31.0 only. It does not establish compatibility with other ONNX Runtime versions or execution providers.
