"""Paired ASR/retrieval experiments with explicit coverage and resumable trials."""
import hashlib
import json
import math
import os
import random
import uuid
import platform
import threading
from importlib.metadata import PackageNotFoundError, version
from collections import Counter
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Callable, TypedDict

import pandas as pd
import numpy as np
import portalocker
from src.knowledge_base import GOLD_MAPPING_VERSION, INTENT_NAMES
from src.metrics import CER_SPACE_POLICY, NORMALIZATION_VERSION, calculate_cer, calculate_raw_wer, calculate_wer, normalize_transcript
from src.minds14_loader import DEFAULT_DATASET_REVISION, MINDS14Loader, MINDS14Sample
from src.sumopod_services import get_embedding, require_asr_model, transcribe_audio
from src.vector_db import (EMBEDDING_PROVENANCE_VERSION,
                          _provider_address as _canonical_provider_address, search_documents)

# Retained for callers displaying the intent inventory; relevance uses document labels.
INTENT_KEYWORDS = {name: [] for name in INTENT_NAMES}
SCHEMA_VERSION = 2
BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = 42
_ACTIVE_CHECKPOINTS: set[str] = set()
_CHECKPOINT_GUARD = threading.Lock()
RESULT_COLUMNS = [
    'trial_id','locale','sample_id','sample_index','intent_class','intent_name','ground_truth',
    'asr_text','wer_score','wer_raw_score','cer_score','reference_words','substitutions','insertions','deletions',
    'hit_rate_text','hit_rate_voice','text_relevance','voice_relevance','status','stage',
    'error_type','error_category','error_message','errors_json','text_documents_json','voice_documents_json',
    'text_document_ids','voice_document_ids','text_document_scores','voice_document_scores',
    'kb_versions','dataset_revision','source_id','audio_sha256','config_hash','normalization_version',
]


class EvaluationResult(TypedDict, total=False):
    locale: str
    sample_id: str
    intent_class: int
    intent_name: str
    ground_truth: str
    asr_text: str
    wer_score: float | None
    hit_rate_text: int | None
    hit_rate_voice: int | None
    status: str


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',',':'), allow_nan=False)


def _hash(value) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _empty_row(sample=None, locale='', index=None):
    sample = sample or {}
    row = {name: None for name in RESULT_COLUMNS}
    row.update(locale=locale,sample_id=sample.get('id',f'{locale}:{index}'),sample_index=index,
        trial_id=f'{locale}:{index}' if index is not None else sample.get('id',''),
        intent_class=sample.get('intent_class'),intent_name=sample.get('intent_name',''),
        ground_truth=sample.get('transcription',''),asr_text='',status='failed',stage='load',
        error_type='',error_category='',error_message='',errors_json='[]',text_documents_json='[]',
        voice_documents_json='[]',text_document_ids='[]',voice_document_ids='[]',
        text_document_scores='[]',voice_document_scores='[]',kb_versions='[]',
        text_relevance='unavailable',voice_relevance='unavailable',
        dataset_revision=sample.get('dataset_revision',''),source_id=sample.get('source_id',''),
        audio_sha256=sample.get('audio_sha256',''),normalization_version=NORMALIZATION_VERSION)
    return row


def check_hit_rate(retrieved_docs: list[dict], intent_class: int) -> bool | None:
    """Label-based hit; None means legacy/unlabeled candidates prevent a verdict."""
    if isinstance(intent_class,bool) or not isinstance(intent_class,int) or not 0 <= intent_class < len(INTENT_NAMES):
        raise ValueError('Invalid intent class')
    if not retrieved_docs:
        return False
    target = INTENT_NAMES[intent_class]
    unknown = False
    for doc in retrieved_docs:
        labels = doc.get('intent_ids') if isinstance(doc,dict) else None
        if not isinstance(labels,list) or not labels or any(x not in INTENT_NAMES for x in labels):
            unknown = True
        elif target in labels:
            return True
    return None if unknown else False


def _run_retrieval(text: str, config: dict | None = None) -> list[dict]:
    config = config or {}
    return search_documents(query_vector=get_embedding(text),top_k=config.get('top_k',3),
                            collection_name=config.get('collection_name'),
                            score_threshold=config.get('score_threshold'))


def evaluate_sample(sample: MINDS14Sample, locale: str, audio_buffer: BytesIO,
                    *, run_config: dict | None = None) -> EvaluationResult:
    """Keep text, ASR and voice stage results independently, including failures."""
    from jiwer import process_words
    config = run_config or {}
    row = _empty_row(sample,locale)
    errors = []
    def error(stage,exc,category='infrastructure'):
        # SDK messages may echo credentials or response bodies; never persist them.
        message = ('ASR returned no words' if category == 'empty_asr' else
                   f'{stage.replace("_"," ").capitalize()} could not be completed')
        errors.append(dict(stage=stage,type=type(exc).__name__,category=category,message=message))
    reference = row['ground_truth']
    if not isinstance(reference,str) or not normalize_transcript(reference):
        error('load',ValueError('Ground truth must contain words'),'dataset')
    else:
        def retrieve(text,kind):
            try:
                docs = _run_retrieval(text,config)
                row[f'{kind}_documents_json'] = _json(docs)
                row[f'{kind}_document_ids'] = _json([d.get('id','') for d in docs])
                row[f'{kind}_document_scores'] = _json([d.get('score') for d in docs])
                hit = check_hit_rate(docs,sample['intent_class'])
                row[f'hit_rate_{kind}'] = int(hit) if hit is not None else None
                row[f'{kind}_relevance'] = 'measured' if hit is not None else 'unlabeled'
                return docs
            except Exception as exc:
                row[f'{kind}_relevance']='error'
                error(f'{kind}_retrieval',exc)
                return []
        # Both branches use the original transcript language; no translation confound.
        text_docs = retrieve(reference,'text')
        voice_docs = []
        try:
            audio_buffer.seek(0)
            transcript = transcribe_audio(audio_buffer)
            if not isinstance(transcript,str):
                raise ValueError('ASR returned a non-string transcript')
            row['asr_text'] = transcript
            row['wer_score'] = calculate_wer(reference,transcript)
            row['wer_raw_score'] = calculate_raw_wer(reference,transcript)
            row['cer_score'] = calculate_cer(reference,transcript)
            normalized_reference, normalized_hypothesis = normalize_transcript(reference),normalize_transcript(transcript)
            word_result = process_words(normalized_reference,normalized_hypothesis)
            row.update(reference_words=len(normalized_reference.split()),substitutions=word_result.substitutions,
                       insertions=word_result.insertions,deletions=word_result.deletions)
            if not normalized_hypothesis:
                row['hit_rate_voice']=0
                row['voice_relevance']='empty_asr'
                error('asr',ValueError('ASR returned no words'),'empty_asr')
            else:
                voice_docs = retrieve(transcript,'voice')
        except Exception as exc:
            error('asr',exc)
        row['kb_versions'] = _json(sorted({d.get('kb_version','') for d in text_docs+voice_docs if d.get('kb_version')}))
        expected = config.get('kb_version')
        if expected and any(d.get('kb_version') != expected for d in text_docs+voice_docs):
            # A collection changed during the run: do not report a hit against mixed KBs.
            row['hit_rate_text']=row['hit_rate_voice']=None
            row['text_relevance']=row['voice_relevance']='error'
            error('retrieval',ValueError('Knowledge-base version changed during evaluation'),'configuration')
    measured = any(row[k] is not None for k in ('wer_score','hit_rate_text','hit_rate_voice'))
    unknown = 'unlabeled' in (row['text_relevance'],row['voice_relevance'])
    row['status'] = ('partial' if errors or unknown else 'success') if measured else 'failed'
    row['stage'] = errors[-1]['stage'] if errors else 'complete'
    row['errors_json'] = _json(errors)
    if errors:
        row.update(error_type=errors[-1]['type'],error_category=errors[-1]['category'],
                   error_message=errors[-1]['message'])
    return row


def select_stratified_indices(loaders: dict, samples_per_locale: int, seed: int = 42) -> dict[str,list[int]]:
    """Shared seeded intent quotas, then independently seeded within-intent samples."""
    if samples_per_locale != -1 and samples_per_locale <= 0:
        raise ValueError('samples_per_locale must be positive or -1')
    grouped={}
    for locale,loader in loaders.items():
        buckets={i:[] for i in range(len(INTENT_NAMES))}
        for index,intent in enumerate(loader.get_intent_classes()):
            buckets[intent].append(index)
        grouped[locale]=buckets
    if samples_per_locale == -1:
        return {locale:list(range(len(loader))) for locale,loader in loaders.items()}
    # The same number from every intent in each locale; cap to common support.
    capacities={i:min(len(buckets[i]) for buckets in grouped.values()) for i in range(len(INTENT_NAMES))}
    target=min(samples_per_locale,sum(capacities.values()))
    if target == 0:
        raise ValueError('Selected locales have no shared intent support')
    order=list(range(len(INTENT_NAMES))); random.Random(seed).shuffle(order)
    quotas=Counter()
    while sum(quotas.values()) < target:
        for intent in order:
            if quotas[intent] < capacities[intent]:
                quotas[intent]+=1
                if sum(quotas.values()) == target:
                    break
    selected={}
    for locale,buckets in grouped.items():
        result=[]
        shuffled={}
        for intent in order:
            indices=list(buckets[intent])
            random.Random(f'{seed}:{locale}:{intent}').shuffle(indices)
            shuffled[intent]=indices[:quotas[intent]]
        while any(shuffled.values()):
            for intent in order:
                if shuffled[intent]:
                    result.append(shuffled[intent].pop())
        selected[locale]=result
    return selected


def _default_run_config():
    from src.config import get_settings
    settings = get_settings()
    return dict(embedding_model=settings.embedding_model,
        embedding_dimensions=settings.embedding_dimensions,
        embedding_send_dimensions=settings.embedding_send_dimensions,
        asr_model=settings.asr_model,chat_model=settings.chat_model,
        embedding_base_url=_canonical_provider_address(settings.provider_address('embedding')),
        asr_base_url=_canonical_provider_address(settings.provider_address('asr')),
        chat_base_url=_canonical_provider_address(settings.provider_address('chat')),
        collection_name=getattr(settings,'qdrant_collection','documents'),
        dataset_revision=settings.dataset_revision,
        normalization_version=NORMALIZATION_VERSION,top_k=settings.retrieval_top_k,
        score_threshold=settings.retrieval_score_threshold)


def _identity(loader,index):
    item=loader.dataset[index]
    audio=item.get('audio') or {}
    audio=audio if isinstance(audio,dict) else {}
    raw=audio.get('bytes')
    if raw is None and audio.get('path'):
        try:
            raw=Path(audio['path']).read_bytes()
        except OSError:
            # Missing audio is a recorded per-trial dataset failure, not a lost run.
            raw=None
    labels=getattr(loader,'intent_labels',INTENT_NAMES)
    intent_name=labels[item['intent_class']]
    return dict(index=index,source_id=str(item.get('id') or item.get('path') or audio.get('path') or f'row-{index}'),
                intent_name=intent_name,intent_class=INTENT_NAMES.index(intent_name),
                transcript_hash=_hash([item.get('transcription'),item.get('intent_class')]),
                audio_sha256=hashlib.sha256(raw).hexdigest() if isinstance(raw,bytes) else None)


def _runtime_versions():
    packages={}
    for name in ('numpy','pandas','jiwer','datasets','soundfile','httpx','qdrant-client','pydantic'):
        try:
            packages[name]=version(name)
        except PackageNotFoundError:
            packages[name]='unavailable'
    return {'python':platform.python_version(),'packages':packages}


def _collection_snapshot(collection_name):
    """Separate adapter seam for offline tests; production always pins a snapshot."""
    from src.vector_db import get_collection_snapshot
    return get_collection_snapshot(collection_name)


def _write_manifest(path,manifest):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    temp.replace(path)


def _load_checkpoint(path,expected_hash):
    rows=[]
    seen=set()
    if not path.exists():
        return rows
    lines=path.read_bytes().splitlines(keepends=True)
    position=0
    for n,line in enumerate(lines):
        try:
            row=json.loads(line)
        except (ValueError,UnicodeDecodeError):
            if n != len(lines)-1 or line.endswith(b'\n'):
                raise ValueError('Checkpoint contains a corrupt complete trial')
            with path.open('r+b') as f:
                f.truncate(position)
            break
        if not isinstance(row,dict):
            raise ValueError('Checkpoint contains a non-object trial')
        if row.get('config_hash') != expected_hash:
            raise ValueError('Checkpoint trial configuration does not match manifest')
        if not set(RESULT_COLUMNS).issubset(row) or row.get('status') not in {'success','partial','failed'}:
            raise ValueError('Checkpoint contains an incomplete trial schema')
        if row.get('trial_id') in seen:
            raise ValueError('Checkpoint contains duplicate trial IDs')
        seen.add(row['trial_id'])
        rows.append(row); position+=len(line)
        if n == len(lines)-1 and not line.endswith(b'\n'):
            # A crash can leave valid JSON without its final newline.
            with path.open('ab') as f:
                f.write(b'\n'); f.flush(); os.fsync(f.fileno())
    return rows


def run_batch_evaluation(locales: list[str] | None = None, samples_per_locale: int = 28,
                         progress_callback: Callable | None = None, *, seed: int = 42,
                         checkpoint_path: str | Path | None = None, resume: bool = False,
                         run_config: dict | None = None,
                         selection_callback: Callable[[int], None] | None = None) -> pd.DataFrame:
    """Hold an OS-backed nonblocking run lock before any billable stage."""
    from src.config import get_settings
    if resume and checkpoint_path is None:
        raise ValueError('Resume requires an explicit checkpoint_path')
    path=Path(checkpoint_path) if checkpoint_path is not None else get_settings().results_dir/uuid.uuid4().hex/'trials.jsonl'
    path.parent.mkdir(parents=True,exist_ok=True)
    key=os.path.normcase(str(path.resolve()))
    # Windows file locks can be reentrant within one process; Streamlit sessions
    # also need a thread-safe registry. The OS lock separately covers processes.
    with _CHECKPOINT_GUARD:
        if key in _ACTIVE_CHECKPOINTS:
            raise RuntimeError('This evaluation checkpoint is already in use')
        _ACTIVE_CHECKPOINTS.add(key)
    try:
        with portalocker.Lock(str(path)+'.lock',mode='a+',timeout=0,
                             flags=portalocker.LOCK_EX|portalocker.LOCK_NB):
            return _run_batch_evaluation(locales,samples_per_locale,progress_callback,seed=seed,
                                         checkpoint_path=path,resume=resume,run_config=run_config,
                                         selection_callback=selection_callback)
    finally:
        with _CHECKPOINT_GUARD:
            _ACTIVE_CHECKPOINTS.discard(key)


def _run_batch_evaluation(locales: list[str] | None = None, samples_per_locale: int = 28,
                         progress_callback: Callable | None = None, *, seed: int = 42,
                         checkpoint_path: str | Path | None = None, resume: bool = False,
                         run_config: dict | None = None,
                         selection_callback: Callable[[int], None] | None = None) -> pd.DataFrame:
    """Run matched intent quotas and checkpoint every attempt before UI updates.

    Resume skips all completed attempts (including recorded errors), never silently
    repeating billable trials. Its dataset/model/KB/selection configuration must match.
    """
    locales=['en-US','en-GB','en-AU'] if locales is None else list(locales)
    if not locales or len(locales)!=len(set(locales)):
        raise ValueError('Choose at least one unique locale')
    if isinstance(samples_per_locale,bool) or not isinstance(samples_per_locale,int) or samples_per_locale == 0 or samples_per_locale < -1:
        raise ValueError('samples_per_locale must be positive or -1')
    if isinstance(seed,bool) or not isinstance(seed,int):
        raise ValueError('Evaluation seed must be an integer')
    if resume and checkpoint_path is None:
        raise ValueError('Resume requires an explicit checkpoint_path')
    config=_default_run_config()
    supplied=dict(run_config or {})
    for key in ('embedding_base_url','asr_base_url','chat_base_url'):
        if key in supplied:
            supplied[key]=_canonical_provider_address(supplied[key])
    allowed=set(config)|{'kb_fingerprint','kb_version','asr_language'}
    if set(supplied)-allowed:
        raise ValueError('Unsupported run_config fields: '+', '.join(sorted(set(supplied)-allowed)))
    for key in ('embedding_model','embedding_dimensions','embedding_send_dimensions','asr_model','chat_model',
                'embedding_base_url','asr_base_url','chat_base_url'):
        if key in supplied and supplied[key] != config[key]:
            raise ValueError(f'{key} must match active service settings')
    if supplied.get('asr_language') not in (None,'auto'):
        raise ValueError('ASR currently uses automatic language detection')
    config.update(supplied)
    if isinstance(config['top_k'],bool) or not isinstance(config['top_k'],int) or not 1 <= config['top_k'] <= 20:
        raise ValueError('top_k must be an integer between 1 and 20')
    threshold=config.get('score_threshold')
    if threshold is not None and (isinstance(threshold,bool) or not isinstance(threshold,(int,float)) or
                                   not math.isfinite(threshold) or not -1 <= threshold <= 1):
        raise ValueError('score_threshold must be a finite number between -1 and 1')
    config['normalization_version']=NORMALIZATION_VERSION
    # Local ASR configuration/credential checks must precede paid text embeddings.
    config['asr_model']=require_asr_model()
    snapshot=_collection_snapshot(config.get('collection_name'))
    if snapshot.get('dimensions') is not None and snapshot['dimensions'] != config['embedding_dimensions']:
        raise ValueError('Knowledge-base embedding dimensions do not match service settings')
    if snapshot.get('embedding_model') and snapshot['embedding_model'] != config['embedding_model']:
        raise ValueError('Knowledge-base embedding model does not match service settings')
    stored_provider=snapshot.get('embedding_base_url')
    if stored_provider is None:
        raise ValueError('Knowledge-base embedding provider is unknown for this route; re-ingest it')
    stored_provider=_canonical_provider_address(stored_provider)
    if stored_provider != config['embedding_base_url']:
        raise ValueError('Knowledge-base embedding provider differs from current service settings; re-ingest it')
    config['embedding_provider_verification']=snapshot.get('embedding_provider_verification','explicit-payload')
    config['embedding_provenance_version']=snapshot.get('embedding_provenance_version',EMBEDDING_PROVENANCE_VERSION)
    if supplied.get('kb_fingerprint') and supplied['kb_fingerprint'] != snapshot['fingerprint']:
        raise ValueError('Knowledge-base fingerprint changed before evaluation')
    config['logical_collection']=snapshot['logical_collection']
    config['collection_name']=snapshot['physical_collection']
    config['kb_fingerprint']=snapshot['fingerprint']
    if supplied.get('kb_version') and supplied['kb_version'] != snapshot.get('kb_version'):
        raise ValueError('Knowledge-base version changed before evaluation')
    if snapshot.get('kb_version'):
        config['kb_version']=snapshot['kb_version']
    loaders={loc:MINDS14Loader(language=loc,revision=config['dataset_revision']) for loc in locales}
    selected=select_stratified_indices(loaders,samples_per_locale,seed)
    identities={loc:[_identity(loaders[loc],i) for i in selected[loc]] for loc in locales}
    effective=dict(config,seed=seed,locales=locales,samples_per_locale=samples_per_locale,
                   selected=identities,schema_version=SCHEMA_VERSION,runtime=_runtime_versions(),
                   gold_mapping_version=GOLD_MAPPING_VERSION,
                   cer_space_policy=CER_SPACE_POLICY,
                   uncertainty={'bootstrap_replicates':BOOTSTRAP_REPLICATES,'seed':BOOTSTRAP_SEED,
                                'unit':'utterance','confidence':0.95})
    config_hash=_hash(effective)
    from src.config import get_settings
    path=Path(checkpoint_path) if checkpoint_path is not None else get_settings().results_dir/uuid.uuid4().hex/'trials.jsonl'
    manifest_path=path.parent/(path.stem+'.manifest.json')
    if not resume and (path.exists() or manifest_path.exists()):
        raise FileExistsError('Checkpoint exists; use resume=True or a different path')
    path.parent.mkdir(parents=True,exist_ok=True)
    if resume:
        if not manifest_path.exists():
            raise ValueError('Checkpoint manifest is missing')
        manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest.get('config_hash') != config_hash:
            raise ValueError('Resume configuration/selection/KB fingerprint mismatch')
        results=_load_checkpoint(path,config_hash)
    else:
        results=[]
        manifest=dict(run_id=uuid.uuid4().hex,created_at=datetime.now(timezone.utc).isoformat(),
                      config_hash=config_hash,config=effective,completed=False,
                      checkpoint_path=str(path),attempted_samples=0)
        _write_manifest(manifest_path,manifest)
    completed={r['trial_id'] for r in results}
    expected={f'{loc}:{index}' for loc in locales for index in selected[loc]}
    if completed-expected:
        raise ValueError('Checkpoint contains trials outside the selected dataset')
    total=len(expected)
    current=len(completed)
    if selection_callback:
        selection_callback(total)
    for locale in locales:
        loader=loaders[locale]
        for index in selected[locale]:
            trial=f'{locale}:{index}'
            if trial in completed:
                continue
            try:
                sample=loader.get_sample(index,decode_audio=False)
                audio=loader.get_audio_bytes(index,sample=sample)
                row=evaluate_sample(sample,locale,audio,run_config=config)
            except Exception as exc:
                identity=next(item for item in identities[locale] if item['index']==index)
                metadata=loader.dataset[index]
                fallback=dict(id=f'{locale}:{index}',transcription=metadata.get('transcription',''),
                    source_id=identity['source_id'],intent_class=identity['intent_class'],
                    intent_name=identity['intent_name'],audio_sha256=identity['audio_sha256'],
                    dataset_revision=config['dataset_revision'])
                row=_empty_row(fallback,locale=locale,index=index)
                message='Dataset sample could not be loaded'
                row.update(error_type=type(exc).__name__,error_category='dataset',error_message=message,
                    errors_json=_json([dict(stage='load',type=type(exc).__name__,category='dataset',message=message)]))
            row.update(trial_id=trial,sample_index=index,config_hash=config_hash)
            with path.open('a',encoding='utf-8') as f:
                f.write(_json(row)+'\n'); f.flush(); os.fsync(f.fileno())
            results.append(row); current+=1
            manifest['attempted_samples']=current
            _write_manifest(manifest_path,manifest)
            if progress_callback:
                # A UI callback failure must propagate, preserving the completed trial.
                progress_callback(current,total,locale,row['sample_id'])
    manifest['completed']=True
    _write_manifest(manifest_path,manifest)
    df=pd.DataFrame(results,columns=RESULT_COLUMNS)
    df.attrs.update(manifest=manifest,checkpoint_path=str(path),manifest_path=str(manifest_path))
    return df


def _bootstrap_ci(values):
    """95% percentile CI resampling utterances; undefined with fewer than 2."""
    values=np.asarray(values,dtype=float)
    values=values[np.isfinite(values)]
    if len(values)<2:
        return float('nan'),float('nan')
    indices=np.random.default_rng(BOOTSTRAP_SEED).integers(0,len(values),(BOOTSTRAP_REPLICATES,len(values)))
    return tuple(float(v) for v in np.percentile(values[indices].mean(axis=1),[2.5,97.5]))


def _wilson_ci(values):
    values=pd.to_numeric(values,errors='coerce').dropna()
    n=len(values)
    if not n:
        return float('nan'),float('nan')
    p=float(values.mean()); z=1.959963984540054
    scale=1+z*z/n
    center=(p+z*z/(2*n))/scale
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/scale
    return max(0.,center-half)*100,min(1.,center+half)*100


def _summarize(df,group):
    columns=[group,'total_samples','attempted_samples','successful_samples','partial_samples','failed_samples',
             'avg_wer_%','raw_avg_wer_%','avg_cer_%','corpus_wer_%','std_wer','std_wer_%','wer_samples','cer_samples',
             'text_hit_rate_%','voice_hit_rate_%','text_measured_samples','voice_measured_samples',
             'text_coverage_%','voice_coverage_%','retrieval_unknown_samples',
             'paired_samples','paired_text_hit_rate_%','paired_voice_hit_rate_%','paired_degradation_pp',
             'paired_degradation_ci_low_pp','paired_degradation_ci_high_pp',
             'avg_wer_ci_low_%','avg_wer_ci_high_%','avg_cer_ci_low_%','avg_cer_ci_high_%','text_hit_ci_low_%','text_hit_ci_high_%',
             'voice_hit_ci_low_%','voice_hit_ci_high_%','corpus_wer_ci_low_%','corpus_wer_ci_high_%']
    if df.empty:
        return pd.DataFrame(columns=columns)
    rows=[]
    for name,data in df.groupby(group,sort=True):
        count=len(data)
        wers=pd.to_numeric(data['wer_score'],errors='coerce')
        cers=pd.to_numeric(data.get('cer_score',pd.Series(float('nan'),index=data.index)),errors='coerce')
        text=pd.to_numeric(data['hit_rate_text'],errors='coerce')
        voice=pd.to_numeric(data['hit_rate_voice'],errors='coerce')
        raw=pd.to_numeric(data.get('wer_raw_score',pd.Series(dtype=float)),errors='coerce')
        words=pd.to_numeric(data.get('reference_words',pd.Series(float('nan'),index=data.index)),errors='coerce')
        edit_rows=sum(pd.to_numeric(data.get(k,pd.Series(float('nan'),index=data.index)),errors='coerce')
                      for k in ('substitutions','insertions','deletions'))
        paired=text.notna() & voice.notna()
        differences=(text[paired]-voice[paired])*100
        paired_low,paired_high=_bootstrap_ci(differences)
        wer_low,wer_high=_bootstrap_ci(wers*100)
        cer_low,cer_high=_bootstrap_ci(cers*100)
        text_low,text_high=_wilson_ci(text)
        voice_low,voice_high=_wilson_ci(voice)
        valid=words.notna() & edit_rows.notna() & words.gt(0)
        corpus_low=corpus_high=float('nan')
        if int(valid.sum())>=2:
            counts=words[valid].to_numpy(); edits=edit_rows[valid].to_numpy()
            indices=np.random.default_rng(BOOTSTRAP_SEED).integers(0,len(counts),(BOOTSTRAP_REPLICATES,len(counts)))
            corpus_low,corpus_high=np.percentile(edits[indices].sum(axis=1)/counts[indices].sum(axis=1)*100,[2.5,97.5])
        status=data.get('status',pd.Series('success',index=data.index))
        unknown=data.get('text_relevance',pd.Series('',index=data.index)).eq('unlabeled') | data.get('voice_relevance',pd.Series('',index=data.index)).eq('unlabeled')
        rows.append({group:name,'total_samples':count,'attempted_samples':count,
            'successful_samples':int(status.eq('success').sum()),'partial_samples':int(status.eq('partial').sum()),
            'failed_samples':int(status.eq('failed').sum()),'avg_wer_%':wers.mean()*100,
            'raw_avg_wer_%':raw.mean()*100,'avg_cer_%':cers.mean()*100,
            'corpus_wer_%':edit_rows[valid].sum()/words[valid].sum()*100 if words[valid].sum()>0 else float('nan'),
            'std_wer':wers.std(),'std_wer_%':wers.std()*100,'wer_samples':int(wers.count()),'cer_samples':int(cers.count()),
            'text_hit_rate_%':text.mean()*100,'voice_hit_rate_%':voice.mean()*100,
            'text_measured_samples':int(text.count()),'voice_measured_samples':int(voice.count()),
            'text_coverage_%':text.count()/count*100,'voice_coverage_%':voice.count()/count*100,
            'retrieval_unknown_samples':int(unknown.sum()),'paired_samples':int(paired.sum()),
            'paired_text_hit_rate_%':text[paired].mean()*100,'paired_voice_hit_rate_%':voice[paired].mean()*100,
            'paired_degradation_pp':differences.mean(),'paired_degradation_ci_low_pp':paired_low,
            'paired_degradation_ci_high_pp':paired_high,'avg_wer_ci_low_%':wer_low,'avg_wer_ci_high_%':wer_high,
            'avg_cer_ci_low_%':cer_low,'avg_cer_ci_high_%':cer_high,
            'text_hit_ci_low_%':text_low,'text_hit_ci_high_%':text_high,
            'voice_hit_ci_low_%':voice_low,'voice_hit_ci_high_%':voice_high,
            'corpus_wer_ci_low_%':corpus_low,'corpus_wer_ci_high_%':corpus_high})
    summary=pd.DataFrame(rows,columns=columns)
    summary.attrs['uncertainty_note']='95% descriptive intervals: Wilson for hits; 1000-replicate seeded utterance bootstrap for WER and paired degradation. Assumes independent utterances; not speaker-controlled causal accent inference.'
    return summary


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    """Macro and corpus WER plus per-metric coverage, never treating errors as zero."""
    return _summarize(df,'locale')


def summarize_by_intent(df: pd.DataFrame) -> pd.DataFrame:
    return _summarize(df,'intent_name')
