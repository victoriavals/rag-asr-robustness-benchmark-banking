# QA and verification

## Offline regression suite

**254 tests passed** on 24 November 2026 using Windows and Python 3.12.10. After installing the development requirements, run the suite from the repository root. [Local setup](../README.md#run-locally)

```powershell
python -B -m pytest -q -p no:cacheprovider
```

The suite ignores the workspace `.env`, isolates output paths, and blocks socket connections and DNS. SumoPod responses are simulated at the HTTP transport, dataset fixtures supply audio and transcripts, and integration tests use the Qdrant in-memory adapter.

| Area | Regression coverage |
|---|---|
| Ingestion and retrieval | Invalid vectors, failed staging and alias activation, previous collection retention, embedding provenance, vector distance and size, and scored retrieval. |
| SumoPod requests | One model key, fixed gateway, request shape, bounded text and audio, safe errors, and source citation validation. |
| Assistant | Passive startup, prompt prefill, progress before the response, retry and edit on the same turn, deduplicated transcription, and edited transcript submission. |
| Evaluation | Seeded shared intent quotas, measured and unknown outcomes, raw and normalized WER, CER, coverage, and paired degradation. |
| Batch safety | One active worker per session, safe stop, saved setup on resume, durable trial records, local inspection, configuration mismatch rejection, and lock coverage across processes. |
| Review and exports | Drafts saved separately by answer, previous single-sample results retained, source evidence and provenance in review CSV, and spreadsheet formula escaping. |

The end-to-end batch test uses the actual Streamlit view, evaluation engine, service wrappers, and local Qdrant adapter. It covers interruption after the first recorded trial, resume, empty-ASR outcomes, and infrastructure failures. An empty ASR response retains 100% normalized WER and a voice retrieval miss. When every ASR request fails, the previous usable benchmark stays available.

## Browser verification

The Streamlit UI was rendered in Chromium through gstack `/browse` using local model, database, and dataset fixtures. A loopback-only proxy restricted outbound requests. The browser checks exercise controls and layout with deterministic data.

| Viewport width | Observed behavior |
|---|---|
| 320 px | Keyboard focus reached visible navigation and chat controls while the sidebar was collapsed. The sampled caption measured 4.81:1 contrast. The assistant and paired summary fit the viewport. |
| 375 px | Assistant, Single Sample, Batch, and Review fit the viewport. Switching from the text candidate to the voice candidate opened a separate draft. Returning to text restored the original notes. Scroll position held through batch start, progress, and completion. |
| 768 px | Single Sample showed the reference and ASR transcripts side by side, followed by both answers and source controls. The layout fit the viewport. |
| 1440 px | The assistant used the configured theme and centered reading area. Two rapid Start clicks created one batch worker. Stop left 3 of 18 trials recorded, and resume completed the same checkpoint with 18 unique trial IDs. |

The voice fixture produced one transcription before **Send question**. Submitting the edited draft then triggered retrieval and a grounded response. The screenshots below illustrate UI behavior with synthetic values.

- [Assistant at 320 px](qa/assistant-mobile.png)
- [Single Sample comparison at 768 px](qa/single-sample-comparison.png)
- [Paired batch summary at 320 px](qa/batch-paired-mobile.png)

Earlier captures document [failed batches](qa/batch-failure-preserved.png), [empty ASR handling](qa/single-empty-asr.png), and [the previous mobile layout](qa/mobile-375.png).

## Cached dataset smoke test

An earlier cached-data run loaded the pinned Arrow datasets for `en-US` (563 samples), `en-GB` (592), and `en-AU` (654) without network access. A cap of 28 per locale selected two rows from each of the 14 intents. The audit also matched the 1,809 cached English WAV files to the Arrow audio bytes.

## Remaining verification

- Run real SumoPod requests to confirm the account's model access, embedding dimensions, ASR quality, latency, and cost. Verify a live Qdrant instance separately.
- Exercise physical microphones, browser permissions, downloaded files on disk, and browsers outside the Chromium checks. AppTest covers the generated export content.
- Verify a clean dependency installation and the hosted Windows and Ubuntu jobs before relying on their results. [Test workflow](../.github/workflows/tests.yml)
- Set access controls, request and cost limits, and retention rules for recordings, transcripts, and results before public hosting.

Use the benchmark protocol to interpret intent hit rates, coverage, and uncertainty. [Benchmark protocol](benchmark_protocol.md)
