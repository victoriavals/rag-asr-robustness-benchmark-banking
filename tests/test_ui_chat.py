"""Chat state regressions with native AppTest and offline service seams."""
from io import BytesIO

import src.tabs.live_demo as live
from test_ui import ui, start, open_research


def calls(ui, kind):
    return [data for name, data in ui.calls if name == kind]


def test_user_bubble_is_rendered_before_any_answer_request(ui, monkeypatch):
    events = []
    message = live.st.chat_message
    process = live._process_query

    def render(role, **kwargs):
        events.append(('render', role))
        return message(role, **kwargs)

    def request(query, history=None, progress=None):
        events.append(('request', query))
        return process(query, history, progress)

    monkeypatch.setattr(live.st, 'chat_message', render)
    monkeypatch.setattr(live, '_process_query', request)
    at = start()
    at.chat_input(key='chat_text_input').set_value('How do I check my balance?').run()
    assert not at.exception
    assert events.index(('render', 'user')) < events.index(('request', 'How do I check my balance?'))
    assert len(at.chat_message) == 2 and len(calls(ui, 'llm')) == 1
    at.run()
    assert len(calls(ui, 'llm')) == 1


def test_retry_updates_same_turn_and_does_not_repeat_on_rerun(ui):
    ui.fail = 'llm'
    at = start()
    at.chat_input(key='chat_text_input').set_value('How do I check my balance?').run()
    turn_id = at.session_state['chat_history'][-1]['turn_id']
    assert at.session_state['chat_history'][-1]['status'] == 'error'
    ui.fail = None
    at.button(key=f'retry_answer_{turn_id}').click().run()
    history = at.session_state['chat_history']
    assert not at.exception and len(history) == 2
    assert history[-1]['turn_id'] == turn_id and history[-1]['attempt'] == 2
    assert history[-1]['status'] == 'success'
    assert calls(ui, 'llm')[-1]['history'] == []
    at.run()
    assert len(calls(ui, 'llm')) == 2


def test_edit_is_local_and_resubmits_the_same_failed_turn(ui):
    ui.fail = 'embedding'
    at = start()
    at.chat_input(key='chat_text_input').set_value('Original failed question').run()
    turn_id = at.session_state['chat_history'][-1]['turn_id']
    before = len(ui.calls)
    at.button(key=f'edit_answer_{turn_id}').click().run()
    at.text_area(key=f'edit_question_{turn_id}').set_value('How do I pay a bill?').run()
    assert len(ui.calls) == before and at.chat_input(key='chat_text_input').disabled
    ui.fail = None
    at.button(key='send_edited_question').click().run()
    assert not at.exception
    history = at.session_state['chat_history']
    assert len(history) == 2 and history[0]['content'] == 'How do I pay a bill?'
    assert history[-1]['status'] == 'success' and history[-1]['attempt'] == 2
    assert calls(ui, 'embedding')[-1]['text'] == 'How do I pay a bill?'


def test_failed_pairs_are_excluded_from_followup_and_model_history(ui):
    history = [
        {'role': 'user', 'content': 'Freeze my card', 'turn_id': 'good', 'status': 'success'},
        {'role': 'assistant', 'content': 'Card policy', 'turn_id': 'good', 'status': 'success'},
        {'role': 'user', 'content': 'Failed loan question', 'turn_id': 'bad', 'status': 'error'},
        {'role': 'assistant', 'content': 'Service failure', 'turn_id': 'bad', 'status': 'error'},
    ]
    bounded = live._bounded_history(history)
    assert [item['content'] for item in bounded] == ['Freeze my card', 'Card policy']
    query = live._retrieval_query('How do I undo that?', history)
    assert 'Freeze my card' in query and 'Failed loan question' not in query


def recorder(monkeypatch):
    audio = BytesIO(b'offline recorded fixture')
    audio.name = 'recording.wav'
    monkeypatch.setattr(live, '_render_audio_input',
        lambda: audio if live.st.session_state.audio_epoch == 0 else None)


def test_voice_is_only_a_draft_until_confirmed_and_uses_edited_text(ui, monkeypatch):
    recorder(monkeypatch)
    at = start()
    assert not at.exception and len(calls(ui, 'asr')) == 1
    assert len(at.chat_message) == 0 and not calls(ui, 'embedding')
    editor = next(item for item in at.text_area if item.label == 'Check or edit the transcript')
    editor.set_value('Bagaimana cara membayar tagihan?').run()
    assert not calls(ui, 'embedding') and len(calls(ui, 'asr')) == 1
    at.button(key='send_voice').click().run()
    assert not at.exception and len(at.chat_message) == 2
    assert calls(ui, 'embedding')[-1]['text'] == 'Bagaimana cara membayar tagihan?'
    assert at.session_state['chat_history'][0]['origin'] == 'voice'
    assert at.session_state['voice_draft'] is None and len(calls(ui, 'asr')) == 1
    assert 'voice_notice' not in at.session_state


def test_retry_transcription_only_updates_draft(ui, monkeypatch):
    recorder(monkeypatch)
    at = start()
    at.button(key='retry_voice').click().run()
    assert not at.exception and len(calls(ui, 'asr')) == 2
    assert not calls(ui, 'embedding') and len(at.chat_message) == 0
    assert next(item for item in at.text_area if item.label == 'Check or edit the transcript').value == 'What is my balance?'
    at.run()
    assert len(calls(ui, 'asr')) == 2


def test_voice_draft_survives_navigation_without_retranscribing(ui, monkeypatch):
    recorder(monkeypatch)
    at = start()
    next(item for item in at.text_area if item.label == 'Check or edit the transcript').set_value('My edited transcript').run()
    open_research(at)
    at.button_group(key='mode_selector').set_value(['Assistant']).run()
    assert not at.exception
    assert next(item for item in at.text_area if item.label == 'Check or edit the transcript').value == 'My edited transcript'
    assert len(calls(ui, 'asr')) == 1 and not calls(ui, 'embedding')


def test_asr_disabled_keeps_chat_and_examples_available(ui, monkeypatch):
    monkeypatch.delenv('ASR_MODEL', raising=False)
    live.get_settings.cache_clear()
    at = start()
    assert not at.exception and ui.calls == []
    assert not at.get('audio_input')
    assert not at.chat_input(key='chat_text_input').disabled
    assert at.button(key='clear_chat').disabled
    at.button(key='chat_example_0').click().run()
    assert ui.calls == [] and not at.chat_message
    assert at.chat_input(key='chat_text_input').value == 'Bagaimana cara mengecek saldo?'
    at.chat_input(key='chat_text_input').set_value('Bagaimana cara mengecek saldo?').run()
    assert not at.exception and len(at.chat_message) == 2
    assert calls(ui, 'embedding')[-1]['text'] == 'Bagaimana cara mengecek saldo?'
    assert not calls(ui, 'asr')


def test_empty_answer_exposes_retry_and_is_not_successful_history(ui, monkeypatch):
    monkeypatch.setattr(live, 'generate_grounded_response', lambda **kwargs: '')
    at = start()
    at.chat_input(key='chat_text_input').set_value('Question').run()
    assert not at.exception
    message = at.session_state['chat_history'][-1]
    assert message['status'] == 'error' and 'No answer' in message['content']
    assert at.button(key=f"retry_answer_{message['turn_id']}")
    assert live._bounded_history(at.session_state['chat_history']) == []


def test_clear_removes_audio_draft_and_does_not_repeat_retained_recording(ui, monkeypatch):
    recorder(monkeypatch)
    at = start()
    at.button(key='clear_chat').click().run()
    assert not at.exception
    assert at.session_state['voice_draft'] is None and at.session_state['chat_history'] == []
    at.run()
    assert len(calls(ui, 'asr')) == 1 and not calls(ui, 'embedding')
