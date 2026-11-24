import hashlib
import json
from collections import Counter
from io import BytesIO
from pathlib import Path
import pandas as pd
import pytest
import portalocker
import os
import subprocess
import sys
import threading
from src import evaluation_engine as mod


class FakeLoader:
    def __init__(self,language='en-US',revision=None):
        self.language=language
        self.revision=revision or mod.DEFAULT_DATASET_REVISION
        self.dataset=[dict(path=f'row-{i}.wav',audio={'bytes':f'wave-{i}'.encode()},
            intent_class=i//2,transcription='check my balance') for i in range(28)]
    def __len__(self): return len(self.dataset)
    def get_intent_classes(self): return [r['intent_class'] for r in self.dataset]
    def get_sample(self,index,decode_audio=True):
        r=self.dataset[index]
        return dict(id=f'{self.language}:{index}',source_id=r['path'],intent_class=r['intent_class'],
            intent_name=mod.INTENT_NAMES[r['intent_class']],transcription=r['transcription'],
            english_transcription=r['transcription'],dataset_revision=self.revision,
            audio_sha256=hashlib.sha256(r['audio']['bytes']).hexdigest(),audio_path='unused')
    def get_audio_bytes(self,index,sample=None):
        b=BytesIO(str(index).encode()); b.name='mock.wav'; return b


@pytest.fixture
def services(monkeypatch):
    monkeypatch.setenv('ASR_MODEL','asr-fixture')
    monkeypatch.setenv('API_KEY','offline-batch-key')
    calls=Counter()
    def embed(text):
        calls['embedding']+=1
        assert text.strip(), 'Empty ASR must not be embedded'
        return [1,0]
    def transcribe(buf):
        calls['asr']+=1; return 'check my balance'
    def search(**kwargs):
        calls['search']+=1
        return [dict(id='labeled-doc',text='bank rules',score=.8,intent_ids=list(mod.INTENT_NAMES),
                     kb_version='kb1',source='test',chunk_index=0)]
    monkeypatch.setattr(mod,'MINDS14Loader',FakeLoader)
    monkeypatch.setattr(mod,'get_embedding',embed)
    monkeypatch.setattr(mod,'transcribe_audio',transcribe)
    monkeypatch.setattr(mod,'search_documents',search)
    monkeypatch.setattr(mod,'_collection_snapshot',lambda *a:dict(fingerprint='fingerprint1',
        physical_collection='immutable1',logical_collection='documents',kb_version='kb1',
        embedding_base_url='https://ai.sumopod.com/v1'))
    return calls


def test_label_hits_never_use_substrings_or_treat_unknown_as_zero():
    assert mod.check_hit_rate([{'text':'Pinjaman has pin','intent_ids':['business_loan']}],9) is False
    assert mod.check_hit_rate([{'text':'Tabungan has bunga','intent_ids':['freeze']}],5) is False
    assert mod.check_hit_rate([{'intent_ids':[]}],4) is None
    assert mod.check_hit_rate(['old plain text'],4) is None
    assert mod.check_hit_rate([],4) is False
    assert mod.check_hit_rate([{'intent_ids':[]},{'intent_ids':['balance']}],4) is True


def test_stratification_equal_quotas_across_locales_and_seed():
    loaders={loc:FakeLoader(loc) for loc in ['en-US','en-GB','en-AU']}
    # Deliberately reorder locale metadata; allocation still matches intent, not prefix.
    loaders['en-AU'].dataset.reverse()
    a=mod.select_stratified_indices(loaders,10,42)
    b=mod.select_stratified_indices(loaders,10,42)
    assert a==b
    quotas=[Counter(loaders[loc].get_intent_classes()[i] for i in indices) for loc,indices in a.items()]
    assert quotas[0]==quotas[1]==quotas[2]
    assert len(quotas[0])==10 and set(quotas[0].values())=={1}
    full=mod.select_stratified_indices(loaders,28,42)
    assert all(len(set(loaders[loc].get_intent_classes()[i] for i in v))==14 for loc,v in full.items())


def test_empty_asr_retained_with_wer_one_voice_zero(tmp_path,monkeypatch,services):
    monkeypatch.setattr(mod,'transcribe_audio',lambda b:'   ')
    df=mod.run_batch_evaluation(['en-US'],2,checkpoint_path=tmp_path/'run.jsonl')
    assert len(df)==2 and df['asr_text'].tolist()==['   ','   ']
    assert df['wer_score'].tolist()==[1,1]
    assert df['hit_rate_voice'].tolist()==[0,0]
    assert df['voice_relevance'].tolist()==['empty_asr','empty_asr']
    assert services['embedding']==2 # text only; voice is explicitly empty
    summary=mod.summarize_results(df).iloc[0]
    assert summary['attempted_samples']==2 and summary['voice_hit_rate_%']==0
    assert summary['voice_coverage_%']==100 and summary['avg_wer_%']==100


def test_infrastructure_failures_are_retained_and_not_measured_zeros(tmp_path,monkeypatch,services):
    def fail(*a,**k): raise RuntimeError('Mock service outage')
    monkeypatch.setattr(mod,'get_embedding',fail)
    monkeypatch.setattr(mod,'transcribe_audio',fail)
    df=mod.run_batch_evaluation(['en-US'],2,checkpoint_path=tmp_path/'run.jsonl')
    assert len(df)==2 and df['status'].eq('failed').all()
    assert df['wer_score'].isna().all() and df['hit_rate_voice'].isna().all()
    summary=mod.summarize_results(df).iloc[0]
    assert summary['failed_samples']==2 and summary['voice_measured_samples']==0
    assert pd.isna(summary['voice_hit_rate_%'])
    assert list(mod.summarize_results(pd.DataFrame()).columns)
    assert mod.summarize_results(pd.DataFrame()).empty


def test_unlabeled_retrieval_exposes_missing_coverage(tmp_path,monkeypatch,services):
    monkeypatch.setattr(mod,'_collection_snapshot',lambda *a:dict(fingerprint='legacy',
        physical_collection='old',logical_collection='documents',kb_version=None,
        embedding_base_url='https://ai.sumopod.com/v1'))
    monkeypatch.setattr(mod,'search_documents',lambda **k:[dict(id='old',text='balance',score=.9,intent_ids=[])])
    df=mod.run_batch_evaluation(['en-US'],2,checkpoint_path=tmp_path/'run.jsonl')
    assert df['hit_rate_text'].isna().all() and df['status'].eq('partial').all()
    summary=mod.summarize_results(df).iloc[0]
    assert summary['text_coverage_%']==0 and summary['retrieval_unknown_samples']==2
    assert summary['wer_samples']==2


def test_resume_skips_completed_paid_trials_and_rejects_changed_config(tmp_path,monkeypatch,services):
    path=tmp_path/'run.jsonl'
    original=mod.run_batch_evaluation(['en-US'],2,checkpoint_path=path)
    counts=services.copy()
    resumed=mod.run_batch_evaluation(['en-US'],2,checkpoint_path=path,resume=True)
    assert len(resumed)==2 and services==counts
    assert original.attrs['manifest']['config_hash']==resumed.attrs['manifest']['config_hash']
    with pytest.raises(ValueError,match='mismatch'):
        mod.run_batch_evaluation(['en-US'],2,seed=99,checkpoint_path=path,resume=True)
    monkeypatch.setattr(mod,'_collection_snapshot',lambda *a:dict(fingerprint='changed-kb',
        physical_collection='immutable1',logical_collection='documents',kb_version='kb1',
        embedding_base_url='https://ai.sumopod.com/v1'))
    with pytest.raises(ValueError,match='mismatch'):
        mod.run_batch_evaluation(['en-US'],2,checkpoint_path=path,resume=True)
    assert services==counts


def test_checkpoint_before_callback_failure_and_resume(tmp_path,services):
    path=tmp_path/'run.jsonl'
    totals = []

    def selected(total):
        assert path.with_suffix('.manifest.json').is_file()
        assert not path.exists() and services['asr'] == 0
        totals.append(total)

    def stop(*a): raise RuntimeError('UI stopped')
    with pytest.raises(RuntimeError,match='UI stopped'):
        mod.run_batch_evaluation(['en-US'],2,progress_callback=stop,checkpoint_path=path,
                                 selection_callback=selected)
    assert len(path.read_text().splitlines())==1 and services['asr']==1
    assert totals == [2]
    df=mod.run_batch_evaluation(['en-US'],2,checkpoint_path=path,resume=True)
    assert len(df)==2 and services['asr']==2


def test_partial_checkpoint_tail_recovered_but_complete_corruption_rejected(tmp_path,services):
    path=tmp_path/'run.jsonl'
    mod.run_batch_evaluation(['en-US'],2,checkpoint_path=path)
    with path.open('ab') as f: f.write(b'{"trial_id":')
    df=mod.run_batch_evaluation(['en-US'],2,checkpoint_path=path,resume=True)
    assert len(df)==2 and len(path.read_text().splitlines())==2
    with path.open('ab') as f: f.write(b'bad complete record\n')
    with pytest.raises(ValueError,match='corrupt'):
        mod.run_batch_evaluation(['en-US'],2,checkpoint_path=path,resume=True)


def test_summary_macro_corpus_and_units():
    rows=[dict(locale='en-US',intent_name='balance',status='success',sample_id='short',wer_score=1,
               wer_raw_score=1,reference_words=1,substitutions=1,insertions=0,deletions=0,hit_rate_text=1,hit_rate_voice=0),
          dict(locale='en-US',intent_name='balance',status='success',sample_id='long',wer_score=0,
               wer_raw_score=0,reference_words=100,substitutions=0,insertions=0,deletions=0,hit_rate_text=1,hit_rate_voice=1)]
    summary=mod.summarize_results(pd.DataFrame(rows)).iloc[0]
    assert summary['avg_wer_%']==50
    assert summary['corpus_wer_%']==pytest.approx(100/101)
    assert summary['std_wer_%']==pytest.approx(summary['std_wer']*100)


@pytest.mark.parametrize('locales,n', [([],2),(['en-US','en-US'],2),(['en-US'],0),(['en-US'],-2),(['en-US'],True)])
def test_invalid_input_fails_before_service_calls(tmp_path,services,locales,n):
    with pytest.raises(ValueError): mod.run_batch_evaluation(locales,n,checkpoint_path=tmp_path/'run.jsonl')
    assert not services


def test_manifest_and_rows_capture_actual_identity_and_retrieval(tmp_path,services):
    df=mod.run_batch_evaluation(['en-US'],2,checkpoint_path=tmp_path/'run.jsonl')
    row=df.iloc[0]
    assert row['dataset_revision']==mod.DEFAULT_DATASET_REVISION
    assert len(row['audio_sha256'])==64 and row['source_id'].endswith('.wav')
    assert json.loads(row['text_document_ids'])==['labeled-doc']
    assert json.loads(row['text_document_scores'])==[.8]
    assert json.loads(row['kb_versions'])==['kb1']
    assert df.attrs['manifest']['completed'] is True
    assert len(df.attrs['manifest']['config']['selected']['en-US'])==2
    assert df.attrs['manifest']['config']['runtime']['python']
    assert df.attrs['manifest']['config']['gold_mapping_version']


def test_safe_errors_never_persist_provider_secrets_or_body(tmp_path,monkeypatch,services):
    secret='sk-sensitive-provider-token'
    def fail(*a,**k): raise RuntimeError(secret+' RESPONSE BODY SECRET')
    monkeypatch.setattr(mod,'transcribe_audio',fail)
    df=mod.run_batch_evaluation(['en-US'],2,checkpoint_path=tmp_path/'run.jsonl')
    assert secret not in df.to_json()
    assert secret not in (tmp_path/'run.jsonl').read_text()
    assert 'RESPONSE BODY SECRET' not in (tmp_path/'run.jsonl').read_text()
    assert df['error_type'].eq('RuntimeError').all()


def test_paired_denominator_excludes_unmatched_infra_failures_and_ci_reproducible():
    df=pd.DataFrame([
        dict(locale='en-US',sample_id='paired',status='success',intent_name='balance',wer_score=.2,
             hit_rate_text=0,hit_rate_voice=1),
        dict(locale='en-US',sample_id='unpaired',status='partial',intent_name='balance',wer_score=.1,
             hit_rate_text=1,hit_rate_voice=None),
        dict(locale='en-US',sample_id='paired2',status='success',intent_name='balance',wer_score=.3,
             hit_rate_text=1,hit_rate_voice=1)])
    a=mod.summarize_results(df).iloc[0]
    b=mod.summarize_results(df).iloc[0]
    assert a['paired_samples']==2
    assert a['paired_text_hit_rate_%']==50 and a['paired_voice_hit_rate_%']==100
    assert a['paired_degradation_pp']==-50
    assert a['text_measured_samples']==3 and a['voice_measured_samples']==2
    assert a['paired_degradation_ci_low_pp']==b['paired_degradation_ci_low_pp']
    assert a['avg_wer_ci_low_%']<=20<=a['avg_wer_ci_high_%']
    assert 0<=a['voice_hit_ci_low_%']<=a['voice_hit_ci_high_%']<=100


def test_batch_freezes_physical_snapshot_and_checks_expected_fingerprint(tmp_path,monkeypatch,services):
    names=[]
    original=mod.search_documents
    def capture(**kwargs):
        names.append(kwargs['collection_name']); return original(**kwargs)
    monkeypatch.setattr(mod,'search_documents',capture)
    df=mod.run_batch_evaluation(['en-US'],2,checkpoint_path=tmp_path/'run.jsonl')
    assert names==['immutable1']*4
    assert df.attrs['manifest']['config']['logical_collection']=='documents'
    assert df.attrs['manifest']['config']['collection_name']=='immutable1'
    with pytest.raises(ValueError,match='fingerprint changed'):
        mod.run_batch_evaluation(['en-US'],2,checkpoint_path=tmp_path/'other.jsonl',
                                 run_config={'kb_fingerprint':'stale'})


def test_load_failure_keeps_known_intent_identity_without_error_body(tmp_path,monkeypatch,services):
    class BadLoader(FakeLoader):
        def get_sample(self,*a,**k): raise ValueError('private-secret-invalid-wav')
    monkeypatch.setattr(mod,'MINDS14Loader',BadLoader)
    df=mod.run_batch_evaluation(['en-US'],2,checkpoint_path=tmp_path/'run.jsonl')
    assert len(df)==2 and df['status'].eq('failed').all()
    assert df['intent_class'].notna().all() and df['ground_truth'].eq('check my balance').all()
    assert df['audio_sha256'].str.len().eq(64).all()
    assert 'private-secret' not in df.to_json()


@pytest.mark.parametrize('config',[{'top_k':0},{'top_k':True},{'score_threshold':float('nan')},
                                   {'score_threshold':2},{'score_threshold':'invalid'}])
def test_retrieval_config_invalid_before_embedding(tmp_path,services,config):
    with pytest.raises(ValueError):
        mod.run_batch_evaluation(['en-US'],2,checkpoint_path=tmp_path/'run.jsonl',run_config=config)
    assert not services


def test_resume_valid_json_without_newline_does_not_join_records(tmp_path,services):
    path=tmp_path/'run.jsonl'
    def stop(*a): raise RuntimeError('Stop after first recorded attempt')
    with pytest.raises(RuntimeError):
        mod.run_batch_evaluation(['en-US'],2,progress_callback=stop,checkpoint_path=path)
    path.write_bytes(path.read_bytes().rstrip(b'\n'))
    df=mod.run_batch_evaluation(['en-US'],2,checkpoint_path=path,resume=True)
    assert len(df)==2
    assert len([json.loads(line) for line in path.read_text().splitlines()])==2


def test_concurrent_checkpoint_run_rejected_before_billable_calls(tmp_path,services,monkeypatch):
    path=tmp_path/'run.jsonl'
    entered=threading.Event(); release=threading.Event(); errors=[]
    original=mod.get_embedding
    def held(text):
        entered.set()
        if not release.wait(20): raise RuntimeError('Test thread was not released')
        return original(text)
    monkeypatch.setattr(mod,'get_embedding',held)
    def first_run():
        try: mod.run_batch_evaluation(['en-US'],2,checkpoint_path=path)
        except BaseException as exc: errors.append(exc)
    worker=threading.Thread(target=first_run)
    worker.start()
    try:
        assert entered.wait(20),'First run never reached its mocked embedding stage'
        before=services.copy()
        with pytest.raises(RuntimeError,match='already in use'):
            mod.run_batch_evaluation(['en-US'],2,checkpoint_path=path)
        assert services==before
    finally:
        release.set(); worker.join(20)
    assert not worker.is_alive() and not errors
    before=services.copy()
    assert len(mod.run_batch_evaluation(['en-US'],2,checkpoint_path=path,resume=True))==2
    assert services==before


def test_separate_process_checkpoint_lock_blocks_paid_calls(tmp_path,services):
    path=tmp_path/'run.jsonl'
    script=("import sys,portalocker; "
            "lock=portalocker.Lock(sys.argv[1],mode='a+',timeout=0,flags=portalocker.LOCK_EX|portalocker.LOCK_NB); "
            "lock.acquire(); print('locked',flush=True); sys.stdin.readline(); lock.release()")
    options={'creationflags':subprocess.CREATE_NO_WINDOW} if os.name=='nt' else {}
    proc=subprocess.Popen([sys.executable,'-B','-c',script,str(path)+'.lock'],
        stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,**options)
    try:
        assert proc.stdout.readline().strip()=='locked'
        with pytest.raises(portalocker.exceptions.LockException):
            mod.run_batch_evaluation(['en-US'],2,checkpoint_path=path)
        assert not services
    finally:
        if proc.poll() is None:
            proc.stdin.write('release\n'); proc.stdin.flush()
            try: proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill(); proc.wait(timeout=5)
        proc.stdin.close(); proc.stdout.close(); proc.stderr.close()


def test_provider_routes_and_request_shape_recorded_and_resume_checked(tmp_path,services,monkeypatch):
    from src.config import Settings
    cfg=Settings(_env_file=None,api_key='offline-secret',asr_model='asr-fixture')
    monkeypatch.setattr('src.config.get_settings',lambda:cfg)
    monkeypatch.setattr(mod,'_collection_snapshot',lambda *a:dict(fingerprint='same',physical_collection='fixed',
        logical_collection='documents',kb_version='kb1',embedding_base_url=cfg.provider_address('embedding'),
        embedding_provider_verification='explicit-metadata-and-payload'))
    path=tmp_path/'run.jsonl'
    df=mod.run_batch_evaluation(['en-US'],2,checkpoint_path=path)
    config=df.attrs['manifest']['config']
    assert config['embedding_base_url']=='https://ai.sumopod.com/v1'
    assert config['asr_base_url']=='https://ai.sumopod.com/v1'
    assert config['chat_base_url']=='https://ai.sumopod.com/v1'
    assert config['embedding_send_dimensions'] is False
    assert config['embedding_provider_verification']=='explicit-metadata-and-payload'
    assert 'offline-secret' not in json.dumps(df.attrs['manifest'])
    count=services.copy()
    changed=Settings(_env_file=None,api_key='offline-secret',asr_model='asr-fixture',embedding_send_dimensions=True)
    monkeypatch.setattr('src.config.get_settings',lambda:changed)
    with pytest.raises(ValueError,match='mismatch'):
        mod.run_batch_evaluation(['en-US'],2,checkpoint_path=path,resume=True)
    assert services==count


def test_sumopod_rejects_unverified_legacy_snapshot_before_paid_calls(tmp_path,services,monkeypatch):
    from src.config import Settings
    cfg=Settings(_env_file=None,api_key='offline-secret',asr_model='asr-fixture')
    monkeypatch.setattr('src.config.get_settings',lambda:cfg)
    monkeypatch.setattr(mod,'_collection_snapshot',lambda *a:dict(fingerprint='unknown',
        physical_collection='old',logical_collection='documents',kb_version='kb1'))
    with pytest.raises(ValueError,match='provider is unknown'):
        mod.run_batch_evaluation(['en-US'],2,checkpoint_path=tmp_path/'run.jsonl')
    assert not services


@pytest.mark.parametrize('missing,expected',[('ASR_MODEL','ASR_MODEL'),('API_KEY','API_KEY')])
def test_asr_local_preflight_blocks_snapshot_and_paid_baseline(tmp_path,services,monkeypatch,missing,expected):
    from src.config import get_settings
    monkeypatch.delenv(missing,raising=False)
    get_settings.cache_clear()
    snapshots=[]
    monkeypatch.setattr(mod,'_collection_snapshot',lambda *a:snapshots.append(a))
    path=tmp_path/'blocked.jsonl'
    with pytest.raises(ValueError,match=expected):
        mod.run_batch_evaluation(['en-US'],2,checkpoint_path=path)
    assert not snapshots and not services
    assert not path.exists()


def test_invalid_count_precedes_asr_preflight(tmp_path,services,monkeypatch):
    calls=[]
    monkeypatch.setattr(mod,'require_asr_model',lambda:calls.append('preflight'))
    with pytest.raises(ValueError,match='positive'):
        mod.run_batch_evaluation(['en-US'],0,checkpoint_path=tmp_path/'blocked.jsonl')
    assert not calls and not services


def test_prior_asr_route_checkpoint_is_incompatible_before_paid_calls(tmp_path,services):
    path=tmp_path/'run.jsonl'
    first=mod.run_batch_evaluation(['en-US'],2,checkpoint_path=path)
    manifest_path=Path(first.attrs['manifest_path'])
    manifest=json.loads(manifest_path.read_text())
    manifest['config']['asr_base_url']='https://previous-provider.example.invalid/v1'
    manifest['config_hash']=mod._hash(manifest['config'])
    manifest_path.write_text(json.dumps(manifest),encoding='utf-8')
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    for row in rows: row['config_hash']=manifest['config_hash']
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows),encoding='utf-8')
    before=services.copy()
    with pytest.raises(ValueError,match='mismatch'):
        mod.run_batch_evaluation(['en-US'],2,checkpoint_path=path,resume=True)
    assert services==before


def test_manifest_runtime_and_routes_are_sumopod_only(tmp_path,services):
    result=mod.run_batch_evaluation(['en-US'],2,checkpoint_path=tmp_path/'run.jsonl')
    config=result.attrs['manifest']['config']
    assert {config[f'{purpose}_base_url'] for purpose in ('embedding','asr','chat')}=={'https://ai.sumopod.com/v1'}
    assert config['asr_model']=='asr-fixture'
    assert {name for name in config if name.endswith('_base_url')}=={'embedding_base_url','asr_base_url','chat_base_url'}
    assert 'httpx' in config['runtime']['packages']
