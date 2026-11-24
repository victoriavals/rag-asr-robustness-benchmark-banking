import pytest
from src.metrics import calculate_cer, calculate_wer, calculate_raw_wer, cosine_similarity_score, normalize_transcript


def test_punctuation_case_apostrophes_and_unicode():
    assert calculate_wer('check my balance', 'Check my balance.') == 0
    assert calculate_wer('dont freeze my card', "Don't freeze my card!") == 0
    assert normalize_transcript('  ＨＥＬＬＯ—world! ') == 'hello world'
    assert calculate_raw_wer('check my balance','Check my balance.') > 0


@pytest.mark.parametrize('reference,hypothesis,expected', [('', '', 0),('', 'one two', 2),
    ('hello there','',1),('hello','hello a b c',3)])
def test_wer_empty_and_insertions(reference,hypothesis,expected):
    assert calculate_wer(reference,hypothesis) == expected


def test_metrics_reject_invalid_values_and_keep_cosine_range():
    with pytest.raises(TypeError): calculate_wer(None,'hello')
    assert cosine_similarity_score([1,0],[-1,0]) == -1
    assert cosine_similarity_score([0,0],[1,0]) == 0
    with pytest.raises(ValueError): cosine_similarity_score([1],[1,0])
    with pytest.raises(ValueError): cosine_similarity_score([float('nan')],[1])


def test_cer_chinese_and_explicit_whitespace_empty_policy():
    assert calculate_wer('\u4f60\u597d\u4e16\u754c','\u4f60\u597d\u4e16\u95f4')==1
    assert calculate_cer('\u4f60\u597d\u4e16\u754c','\u4f60\u597d\u4e16\u95f4')==.25
    assert calculate_cer('hello world','helloworld')==0
    assert calculate_cer('','')==0
    assert calculate_cer('','a b')==2
    assert calculate_cer('abc','')==1
