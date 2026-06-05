"""Tests for OllamaClient: round-robin rotation, chat, list_models."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from ollama_symphony import WorkflowConfig, OllamaClient, OllamaConnectionError


def _config(hosts=None, **kwargs):
    return WorkflowConfig(
        ollama_hosts=hosts or ["http://host1:11434"],
        **kwargs,
    )


# ---------------------------------------------------------------------------
# _next_host — round-robin
# ---------------------------------------------------------------------------

def test_next_host_single():
    client = OllamaClient(_config(hosts=["http://a:11434"]))
    assert client._next_host() == "http://a:11434"
    assert client._next_host() == "http://a:11434"


def test_next_host_rotates():
    hosts = ["http://a:11434", "http://b:11434", "http://c:11434"]
    client = OllamaClient(_config(hosts=hosts))
    assert client._next_host() == "http://a:11434"
    assert client._next_host() == "http://b:11434"
    assert client._next_host() == "http://c:11434"
    assert client._next_host() == "http://a:11434"  # wraps


# ---------------------------------------------------------------------------
# chat — success path
# ---------------------------------------------------------------------------

def test_chat_success_model_dump():
    cfg = _config()
    oc = OllamaClient(cfg)

    fake_response = MagicMock()
    fake_response.model_dump.return_value = {"message": {"role": "assistant", "content": "hi"}}

    mock_ollama_client = MagicMock()
    mock_ollama_client.chat.return_value = fake_response

    with patch("ollama_symphony.OllamaClient.chat", wraps=oc.chat):
        with patch("ollama.Client", return_value=mock_ollama_client):
            result = oc.chat(messages=[{"role": "user", "content": "hello"}], tools=[])

    assert result == {"message": {"role": "assistant", "content": "hi"}}


def test_chat_success_dict_response():
    cfg = _config()
    oc = OllamaClient(cfg)

    fake_response = {"message": {"role": "assistant", "content": "ok"}}

    mock_ollama_client = MagicMock()
    mock_ollama_client.chat.return_value = fake_response

    with patch("ollama.Client", return_value=mock_ollama_client):
        result = oc.chat(messages=[], tools=[])

    assert result == fake_response


def test_chat_passes_options():
    cfg = _config(ollama_temperature=0.5, ollama_num_ctx=4096)
    oc = OllamaClient(cfg)

    mock_ollama_client = MagicMock()
    mock_ollama_client.chat.return_value = {"message": {}}

    with patch("ollama.Client", return_value=mock_ollama_client):
        oc.chat(messages=[], tools=[])

    call_kwargs = mock_ollama_client.chat.call_args[1]
    opts = call_kwargs["options"]
    assert opts["temperature"] == 0.5
    assert opts["num_ctx"] == 4096


# ---------------------------------------------------------------------------
# chat — fallback on failure
# ---------------------------------------------------------------------------

def test_chat_falls_back_to_second_host():
    cfg = _config(hosts=["http://bad:11434", "http://good:11434"])
    oc = OllamaClient(cfg)

    good_response = {"message": {"content": "pong"}}

    def client_factory(host):
        mock = MagicMock()
        if "bad" in host:
            mock.chat.side_effect = ConnectionError("refused")
        else:
            mock.chat.return_value = good_response
        return mock

    with patch("ollama.Client", side_effect=client_factory):
        result = oc.chat(messages=[], tools=[])

    assert result == good_response


def test_chat_raises_when_all_hosts_fail():
    cfg = _config(hosts=["http://a:11434", "http://b:11434"])
    oc = OllamaClient(cfg)

    mock_ollama_client = MagicMock()
    mock_ollama_client.chat.side_effect = ConnectionError("down")

    with patch("ollama.Client", return_value=mock_ollama_client):
        with pytest.raises(OllamaConnectionError, match="All Ollama hosts unreachable"):
            oc.chat(messages=[], tools=[])


def test_chat_tries_each_host_exactly_once():
    hosts = ["http://a:11434", "http://b:11434", "http://c:11434"]
    cfg = _config(hosts=hosts)
    oc = OllamaClient(cfg)

    created_hosts = []

    def client_factory(host):
        created_hosts.append(host)
        mock = MagicMock()
        mock.chat.side_effect = RuntimeError("fail")
        return mock

    with patch("ollama.Client", side_effect=client_factory):
        with pytest.raises(OllamaConnectionError):
            oc.chat(messages=[], tools=[])

    assert created_hosts == hosts


# ---------------------------------------------------------------------------
# list_models
# ---------------------------------------------------------------------------

def test_list_models_success():
    cfg = _config()
    oc = OllamaClient(cfg)

    mock_client = MagicMock()
    mock_client.list.return_value = {
        "models": [{"name": "llama3"}, {"name": "qwen2"}, {"name": ""}]
    }

    with patch("ollama.Client", return_value=mock_client):
        models = oc.list_models()

    assert models == ["llama3", "qwen2"]


def test_list_models_empty():
    cfg = _config()
    oc = OllamaClient(cfg)

    mock_client = MagicMock()
    mock_client.list.return_value = {"models": []}

    with patch("ollama.Client", return_value=mock_client):
        models = oc.list_models()

    assert models == []


def test_list_models_exception_returns_empty():
    cfg = _config()
    oc = OllamaClient(cfg)

    mock_client = MagicMock()
    mock_client.list.side_effect = ConnectionError("unreachable")

    with patch("ollama.Client", return_value=mock_client):
        models = oc.list_models()

    assert models == []


def test_list_models_non_dict_response_returns_empty():
    cfg = _config()
    oc = OllamaClient(cfg)

    mock_client = MagicMock()
    mock_client.list.return_value = MagicMock()  # not a dict

    with patch("ollama.Client", return_value=mock_client):
        models = oc.list_models()

    assert models == []
