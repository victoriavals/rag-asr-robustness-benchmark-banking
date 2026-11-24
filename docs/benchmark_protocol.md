# Benchmark protocol

## What is measured

Each selected MINDS-14 audio sample is evaluated twice against the same physical collection in Qdrant. The text branch submits the original reference transcript. The voice branch transcribes the audio through SumoPod and submits that transcript. Both branches use the same embedding model, retrieval depth, and score threshold. Any displayed English translation is for inspection only.

**Single Sample** displays both transcripts, word and character error rates, retrieved passages, and the two generated answers. **Batch** measures transcription and retrieval across multiple trials. Answer generation is available in Single Sample and the Assistant.

Batch starts with a cap of 28 samples per locale across `en-US`, `en-GB`, and `en-AU`. Seeded sampling sets the same intent quotas for all selected locales and selects the rows within each intent using a locale-specific random stream. The common supply of each intent can lower the final count. With fewer than 14 samples per locale, only a subset of intents can appear. **Use full datasets** takes every row and preserves each locale's original intent distribution.

The default MINDS-14 revision is `40ce77cb32a384e4d50a568e1ec39ac804019d33`. Reproduce a selection with the same revision, locale list, mode, cap, and seed. Cached WAV files live under `data/minds14/<revision>/` and are checked against the source audio hash before reuse. Locale groups can differ in speakers and content, so comparisons describe the sampled groups.

## Relevance labels

The demonstration knowledge base has 12 service sections and 14 intent labels. Transfer and direct debit share one section. Withdrawal and deposit share another. Labels are assigned using exact known chapter headings in `src/knowledge_base.py`. Other text stays unlabelled. Ingestion omits title-only chunks and uses deterministic IDs and content hashes.

Intent Hit@k is 1 when any returned candidate carries the target intent and 0 when retrieval is empty or all returned candidates have known, nonmatching labels. If candidates include unknown labels and none matches, the verdict is unknown. The app defaults to `RETRIEVAL_TOP_K=3` and accepts an optional `RETRIEVAL_SCORE_THRESHOLD`. Independent question and passage annotations are needed to measure question-level relevance.

## Transcription metrics

Normalized word error rate (WER) uses shared, versioned normalization. The pipeline applies Unicode NFKC and case folding, joins words across apostrophes, replaces other punctuation with spaces, and collapses whitespace. Numerals and their spoken forms remain distinct. The engine also records raw WER without that normalization. WER can exceed 100% when ASR inserts words.

Character error rate (CER) uses the same normalization and omits whitespace. This also supports scripts with limited use of spaces between words. The single-sample cosine similarity ranges from −1 to 1 and describes similarity between the two embeddings.

An empty reference after normalization is an invalid sample. An ASR response with no normalized words against a valid reference counts as 100% normalized WER and a measured voice retrieval miss. The voice branch skips the embedding request for empty input. A service error leaves the failed stage unmeasured and preserves available measurements from other stages.

Reports distinguish mean utterance WER from corpus WER, which divides the total word edits by the total reference words. Inspect the number of measured trials and coverage alongside percentages.

**Paired results by locale** uses only trials with measured retrieval in both branches. ASR degradation is the paired text hit rate minus the paired voice hit rate, in percentage points. A positive value means the voice branch had fewer hits on the matched trials. If text hits 2 of 3 trials and voice hits its only measured trial, the paired comparison uses that one shared trial. When text missed that trial, the paired result is 0% text, 100% voice, and −100 percentage points. The independent text rate of 66.7% belongs in **Coverage and full summary**.

Independent hit-rate intervals use Wilson estimates. WER, CER, and paired-degradation intervals use 1,000 seeded bootstrap resamples of utterances. The bootstrap interval needs at least two measured values for its metric. Metrics with no measured trials display as unavailable. Interpret intervals alongside coverage and trial failures. The resampling assumes independent utterances, and multiple utterances from the same speaker can violate that assumption.

## Provenance, persistence and answer review

The manifest binds source IDs and audio hashes to the dataset revision, locale selection, model routes and settings, physical knowledge base and fingerprint, normalization, and runtime versions. Every trial has a status, stage, safe error category, retrieved IDs and scores, and its measured values. The results and exports may include transcripts and retrieved policy passages.

The engine flushes each JSONL row before reporting progress. **Stop after current trial** stops batch processing at a checkpoint. **Inspect saved checkpoint** reads the local manifest and recorded trials. Resume validates compatibility and skips recorded attempts, including partial or failed ones. It can repair an incomplete final line after an interruption and rejects corrupt complete records. A request interrupted before its trial was recorded can be repeated. [How to inspect and resume a batch](../README.md#checkpoints-and-resume)

**Review** captures human scores for source support, factual correctness, and relevance on a 0–2 scale. It keeps a draft per answer. Saved reviews live in the Streamlit session. The CSV includes the observed passages, reviewer notes, and the originating experiment or chat turn, with sample, locale, and branch when available. Judge the actual question and cited passages and use independent domain evidence when needed.

Keep the manifest alongside any reported metrics. Re-run older benchmarks under this protocol before comparing them with the current results.
