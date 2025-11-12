# Offline Dataset Generation Runs

| Timestamp (UTC) | Run ID | Dataset Tag | Generator Model | Seed | Prompt Template | Tasks Generated | Tasks Filtered | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2025-11-02T23:23:40Z | subset20-20251102T2323 | subset20 | qwen2:1.5b-instruct-q4_0 | 42 | offline_dataset.default_json_prompt | 44 | 64 | Processed 18 docs (2 generator errors); analyzer mean citation coverage 0.72 with 8/8 refs >=0.2; flagged 2510.26771 as problematic (RICH in MATH symbols) |
| 2025-11-06T15:33:41Z | cog-psych-20251106T1533 | cog_psych_2025_run01 | qwen2.5:7b-instruct-q4_K_M | 42 | offline_dataset.default_json_prompt | 392 | 61 | Resume run (max_chunks_per_document=4); 95 docs succeeded, 4 skipped, 61 retried+failed (mostly Ollama restarts). Invalid escape sanitizer recovered 2 payloads. |
