> Author: Naufal Firdaus
>
> Project: UAS Deep Learning — AI NLP and RAG Systems

# Banking RAG and ASR Robustness Benchmark

A Streamlit banking-policy assistant and a paired speech and retrieval benchmark using MINDS-14. SumoPod provides chat, embeddings, and speech transcription. Qdrant stores and searches the policy vectors.

The Research Lab compares a reference transcript and its automatic speech recognition (ASR) transcript against the same Indonesian policy knowledge base. Retrieval scores count matches to chapter intent labels. Human reviewers score the generated answers separately. [Benchmark protocol](docs/benchmark_protocol.md)

## Run locally

### 1. Install dependencies

Use Python 3.12 and run these PowerShell commands from the repository root.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
if (-not (Test-Path -LiteralPath '.env')) { Copy-Item -LiteralPath '.env.example' -Destination '.env' }
```

### 2. Configure SumoPod and Qdrant

Set `API_KEY` in `.env` to your SumoPod key. The app uses that key for all model requests through `https://ai.sumopod.com/v1`. Restart Streamlit after changing settings. [Environment template](.env.example)

| Setting | Default or example | Purpose |
|---|---|---|
| `MODEL` | `glm-5.3-flash` | Chat model. `CHAT_MODEL` takes precedence when both names are set. |
| `LLM_TEMPERATURE`, `CHAT_MAX_TOKENS` | `0.7`, `1024` | Chat sampling temperature and maximum output tokens. |
| `EMBEDDING_MODEL`, `EMBEDDING_DIMENSIONS` | `text-embedding-3-small`, `1536` | Model and vector size used for indexing and retrieval. |
| `EMBEDDING_SEND_DIMENSIONS` | `false` | Enable if the model accepts the optional `dimensions` parameter. The app validates the returned vector size in either case. |
| `ASR_MODEL` | Empty | Exact transcription model ID enabled for your account and the `/v1/audio/transcriptions` endpoint. |
| `QDRANT_URL`, `QDRANT_COLLECTION` | `http://localhost:6333`, `test-ai-nlp` | URL and logical collection from the example. The code defaults to `documents` if the collection setting is omitted. |
| `QDRANT_API_KEY` | Empty | Set separately for an authenticated Qdrant cluster. Use an HTTP(S) URL without embedded credentials. |

With `ASR_MODEL` blank, you can use text chat, preview samples in **Single Sample**, and inspect saved batch checkpoints. Voice input and speech experiments require a configured transcription model. Check the models and output dimensions enabled for your SumoPod account before running live requests. [SumoPod model selection](https://docs.sumopod.com/guides/ai/change-model)

### 3. Start Qdrant, index the policy, and launch Streamlit

For local Qdrant, use Docker to bind port 6333 to loopback and store data in `qdrant_storage/`. If you already have a Qdrant server, set `QDRANT_URL` and start at the ingestion command.

```powershell
docker run -d --name banking-qdrant -p 127.0.0.1:6333:6333 -v "${PWD}/qdrant_storage:/qdrant/storage" qdrant/qdrant
```

Restart an existing local container with `docker start banking-qdrant`. Once Qdrant is ready, index the policy and launch the app.

```powershell
.\.venv\Scripts\python.exe ingest_data.py
.\.venv\Scripts\python.exe -m streamlit run main.py
```

Ingestion sends policy chunks to SumoPod for embedding. It indexes the bundled demonstration policy in `data/bank_policy.txt` by default and accepts a text file path as its first argument.

Open the local URL printed by Streamlit. The app binds to `127.0.0.1` and caps uploads at 25 MiB by default. Git ignores `.env`, the dataset audio cache, the default `results/` directory, and Qdrant storage. [Streamlit configuration](.streamlit/config.toml)

## Application behavior

| Workspace | How to use it |
|---|---|
| **Assistant** | Type a question or choose an example to fill the composer, then send it. Open **Sources** to read the passages supporting the answer. **Retry answer** and **Edit question** update the failed turn. With voice enabled, open **Voice input** and record a question. Transcription sends audio to SumoPod. Review or edit the draft, then select **Send question** to start retrieval. |
| **Research Lab → Single Sample** | Pick a dataset locale and select **Load dataset**. Choose a sample, listen to its recording, and select **Run experiment** to compare the reference and ASR transcripts and their answers. The last successful result stays visible with its sample and timestamp if a later attempt fails. |
| **Research Lab → Batch** | Choose locales and a sample cap, then select **Start batch evaluation**. The default is 28 samples per locale across `en-US`, `en-GB`, and `en-AU`. The main comparison uses trials with retrieval scores for both inputs. **Coverage and full summary** shows each input's denominator. |
| **Research Lab → Review** | Choose the most recent successful chat answer or one of the two answers from the last successful single-sample experiment. Score source support, factual correctness, and relevance before selecting **Save review**. Each answer has its own draft. Use **Download reviews CSV** to keep a copy outside the session. |

Model, database, and dataset requests start on explicit actions. Select **Refresh database status** to check Qdrant. Loading a dataset or starting a batch can download uncached MINDS-14 data. Each batch trial can send up to one transcription request and two embedding requests.

These screenshots show the interface with local fixtures. [Mobile assistant](docs/qa/assistant-mobile.png) · [Single-sample comparison](docs/qa/single-sample-comparison.png)

### Safe ingestion and existing collections

Ingestion validates the input and builds the embeddings before creating a staging collection. After upsert and point-count validation, one alias update activates the new collection through `<QDRANT_COLLECTION>__active`.

Reads resolve the active alias first and fall back to the legacy collection when the alias is absent. Failed staging collections and previously active collections remain available for inspection. The CLI returns a nonzero exit code on failure.

If the sidebar reports incompatible embedding provenance, model, dimension, or distance, check your embedding settings and rerun ingestion. This also adds stable IDs, intent labels, and knowledge-base version metadata to a legacy index. Select **Refresh database status** after the new collection activates.

### Checkpoints and resume

Each run uses a folder under `RESULTS_DIR`, which defaults to `results/`. Keep `trials.jsonl` and `trials.manifest.json` together for recovery. Trial records are flushed before progress updates. The manifest stores the dataset revision, selected IDs and audio hashes, model settings, and knowledge-base snapshot. For analysis, download the trial CSV and manifest from the results panel. The current run and **Previous valid results** have separate export buttons.

1. An active batch keeps running when you switch views in the same session. Once trials are running, select **Stop after current trial** to pause at a checkpoint.
2. In a new session, open **Advanced settings and recovery** in **Batch**, check **Resume an existing run**, enter the `trials.jsonl` path on the Streamlit machine, and select **Inspect saved checkpoint**. Inspection reads the local run files and works without a model key.
3. Select **Resume pending trials with saved setup** for an interrupted run. Resume uses the saved locales, sample cap, and seed, checks compatibility, and skips all recorded trials, including failures.

Use **Start a new replication** to repeat recorded trials deliberately. Start a new run after changing the dataset revision, model settings, sample selection, or knowledge base. Concurrent runs on one checkpoint are blocked. After an abrupt interruption, a request accepted by the provider before its trial was recorded may run again. [Checkpoint and measurement details](docs/benchmark_protocol.md#provenance-persistence-and-answer-review)

The dataset is pinned to an immutable revision. Set `DATASET_REVISION` to a full commit SHA to choose another revision. If you export `HF_HUB_OFFLINE=1` or `HF_DATASETS_OFFLINE=1` in the shell, the loader requires an existing Arrow cache for the selected locale and revision. A speech experiment still uses the configured SumoPod and Qdrant services.

## Offline verification

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider
```

The tests use isolated settings and block socket connections and DNS. They exercise Streamlit AppTest, simulated SumoPod HTTP responses, and Qdrant in memory. The latest local run passed **254 tests**. [QA evidence and coverage](docs/QA.md)

Dependencies are pinned in `requirements.txt` and `requirements-dev.txt`. To refresh the pins after a deliberate upgrade, run `python scripts/lock_dependencies.py` in the tested environment and rerun the suite. The workflow defines Windows and Ubuntu jobs using Python 3.12. [CI workflow](.github/workflows/tests.yml)

## Source layout

- `main.py` and `src/tabs/` implement navigation, chat, transcript drafts, the batch worker, and manual review.
- `src/config.py` and `src/sumopod_services.py` handle validated settings and model requests.
- `ingest_data.py`, `src/knowledge_base.py`, and `src/vector_db.py` build and query the policy index.
- `src/minds14_loader.py`, `src/metrics.py`, and `src/evaluation_engine.py` handle dataset loading, scoring, paired evaluation, and checkpoints.

Code is available under the [MIT License](LICENSE). MINDS-14 is supplied separately. Consult its dataset card for usage terms and provenance. [PolyAI/minds14](https://huggingface.co/datasets/PolyAI/minds14)
