"""All regression tests are offline, including accidental SDK/network calls."""
import socket
import pytest


@pytest.fixture(autouse=True)
def single_selection_app_test(monkeypatch):
    from streamlit.testing.v1.element_tree import ButtonGroup

    original = ButtonGroup.indices.fget

    def indices(group):
        value = group.value
        if value is None:
            return []
        if isinstance(value, str):
            return [group.options.index(group.format_func(value))]
        return original(group)

    # Streamlit 1.52.1 AppTest assumes a list even for single-selection groups.
    monkeypatch.setattr(ButtonGroup, "indices", property(indices))


@pytest.fixture
def settle_batch():
    def settle(at):
        controller = at.session_state["_batch_controller"]
        controller.join(timeout=15)
        assert not controller.snapshot().busy
        at.run()
        assert not at.exception

    return settle

@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError('Network access is prohibited in the offline regression suite')
    monkeypatch.setattr(socket.socket, 'connect', blocked)
    monkeypatch.setattr(socket.socket, 'connect_ex', blocked)
    monkeypatch.setattr(socket, 'getaddrinfo', blocked)
    monkeypatch.setenv('HF_HUB_OFFLINE', '1')
    monkeypatch.setenv('HF_DATASETS_OFFLINE', '1')
    monkeypatch.setenv('PYTHONDONTWRITEBYTECODE', '1')

@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch, tmp_path):
    from src.config import Settings, get_settings
    # Ignore the user's .env and clear aliases in this test process only.
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)
    for name in ('CHAT_MODEL', 'MODEL', 'MINDS14_REVISION'):
        monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()
    monkeypatch.setattr(Settings, 'model_config', {**Settings.model_config, 'env_file': None})
    monkeypatch.setenv('RESULTS_DIR', str(tmp_path / 'results'))
    yield
    get_settings.cache_clear()
