from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
import hashlib
import numpy as np
import pytest
import soundfile as sf
from src import minds14_loader as mod


def audio(value):
    b=BytesIO(); sf.write(b,np.full(80,value),8000,format='WAV'); return b.getvalue()


class Rows(list):
    features={'intent_class':SimpleNamespace(names=list(mod.INTENT_NAMES))}
    def __getitem__(self,key):
        if key=='intent_class': return [r[key] for r in self]
        return super().__getitem__(key)


def loader(tmp_path,raw=None,intent=4):
    l=mod.MINDS14Loader(cache_dir=str(tmp_path))
    l._dataset=Rows([dict(path='stable.wav',audio={'bytes':raw or audio(.1)},
        intent_class=intent,transcription='check balance',english_transcription='check balance')])
    return l


def test_cache_follows_audio_content_and_repairs_corruption(tmp_path):
    l=loader(tmp_path)
    a=l.get_sample(0)
    l._dataset[0]['audio']['bytes']=audio(.8)
    l._dataset[0]['transcription']='changed transcript'
    b=l.get_sample(0)
    assert a['id']==b['id']
    assert a['audio_path']!=b['audio_path']
    assert b['transcription']=='changed transcript'
    assert np.mean(b['audio_array'])==pytest.approx(.8,abs=.001)
    Path(b['audio_path']).write_bytes(b'corrupt cache')
    repaired=l.get_sample(0)
    assert hashlib.sha256(Path(repaired['audio_path']).read_bytes()).hexdigest()==repaired['audio_sha256']


def test_bytes_retrieval_does_not_decode_twice(tmp_path,monkeypatch):
    l=loader(tmp_path); calls=[]
    original=mod.sf.read
    monkeypatch.setattr(mod.sf,'read',lambda *a,**k:(calls.append(1),original(*a,**k))[1])
    sample=l.get_sample(0,decode_audio=False)
    buf=l.get_audio_bytes(0,sample=sample)
    assert not calls
    assert buf.tell()==0 and buf.name.endswith('.wav')
    l.get_sample(0)
    assert len(calls)==1


def test_metadata_class_order_is_respected(tmp_path):
    l=loader(tmp_path,intent=0)
    l._dataset.features={'intent_class':SimpleNamespace(names=list(reversed(mod.INTENT_NAMES)))}
    assert l.get_sample(0)['intent_name']=='pay_bill'
    assert l.get_intent_classes()==[13]


@pytest.mark.parametrize('intent',[-1,14])
def test_invalid_intent_and_indices_rejected(tmp_path,intent):
    l=loader(tmp_path,intent=intent)
    with pytest.raises(ValueError): l.get_sample(0)
    with pytest.raises(IndexError): l.get_sample(-1)
    with pytest.raises(IndexError): l.get_sample(1)


def test_offline_cache_missing_does_not_attempt_download(tmp_path,monkeypatch):
    monkeypatch.setenv('HF_DATASETS_CACHE',str(tmp_path/'absent'))
    monkeypatch.setattr(mod,'load_dataset',lambda *a,**k:pytest.fail('Offline loading must not call hub loader'))
    with pytest.raises(FileNotFoundError,match='cached offline'): loader(tmp_path).load()


def test_revision_and_raw_audio_validation(tmp_path):
    with pytest.raises(ValueError): mod.MINDS14Loader(revision='main')
    with pytest.raises(ValueError): loader(tmp_path,raw=b'bad audio').get_sample(0)
