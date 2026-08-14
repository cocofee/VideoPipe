import numpy as np
import pytest

from realtime.vlm_utils import OpenAIVLMAssistant, VLMRateLimitError
import realtime.vlm_utils as vlm_utils


class _Response:
    def __init__(self, status_code, data=None):
        self.status_code = status_code
        self._data = data or {}

    def json(self):
        return self._data


def test_openai_proxy_chat_payload_and_output_text(monkeypatch):
    calls = []

    def _post(url, **kwargs):
        calls.append((url, kwargs))
        return _Response(200, {"choices": [{"message": {"content": "105\n0.91\nclear"}}]})

    monkeypatch.setattr(vlm_utils.requests, "post", _post)
    assistant = OpenAIVLMAssistant(
        "fake-openai-key", "gpt-4.1-mini", "https://proxy.example/v1"
    )

    result = assistant.call(np.zeros((20, 20, 3), dtype=np.uint8), "read the bib")

    assert result == "105\n0.91\nclear"
    assert calls[0][0] == "https://proxy.example/v1/chat/completions"
    request = calls[0][1]
    assert request["headers"]["Authorization"] == "Bearer fake-openai-key"
    assert request["json"]["model"] == "gpt-4.1-mini"
    content = request["json"]["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "read the bib"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_openai_proxy_full_url_is_not_duplicated(monkeypatch):
    monkeypatch.setattr(
        vlm_utils.requests,
        "post",
        lambda *args, **kwargs: _Response(
            200,
            {"choices": [{"message": {"content": "23"}}]},
        ),
    )
    calls = []
    monkeypatch.setattr(
        vlm_utils.requests,
        "post",
        lambda url, **kwargs: (calls.append(url) or _Response(200, {"choices": [{"message": {"content": "23"}}]})),
    )
    assistant = OpenAIVLMAssistant(
        "fake-openai-key", base_url="https://proxy.example/v1/chat/completions"
    )

    assert assistant.call(np.ones((10, 10, 3), dtype=np.uint8), "read") == "23"
    assert calls == ["https://proxy.example/v1/chat/completions"]


def test_openai_rate_limit_raises_without_logging_key(monkeypatch):
    monkeypatch.setattr(vlm_utils.requests, "post", lambda *args, **kwargs: _Response(429))
    assistant = OpenAIVLMAssistant("fake-openai-key")

    with pytest.raises(VLMRateLimitError):
        assistant.call(np.ones((10, 10, 3), dtype=np.uint8), "read")


def test_openai_ocr_bib_reuses_shared_parser(monkeypatch):
    monkeypatch.setattr(
        vlm_utils.requests,
        "post",
        lambda *args, **kwargs: _Response(200, {"choices": [{"message": {"content": "A123\n0.88\nclear"}}]}),
    )
    assistant = OpenAIVLMAssistant("fake-openai-key")

    assert assistant.ocr_bib(np.ones((10, 10, 3), dtype=np.uint8)) == ("A123", 0.88, "clear")


def test_parse_ocr_result_ignores_reasoning_think_block():
    assert vlm_utils.parse_ocr_result(
        "<think>**Confirming bib number as 669**</think>\n\n"
        "669\n0.82\nThree visible digits"
    ) == ("669", 0.82, "Three visible digits")


def test_parse_ocr_result_rejects_free_form_digits():
    assert vlm_utils.parse_ocr_result(
        "The bib number is 669\n0.90\nclear"
    ) == (None, 0.90, "clear")


def test_openai_call_uses_reasoning_when_content_is_empty(monkeypatch):
    monkeypatch.setattr(
        vlm_utils.requests,
        "post",
        lambda *args, **kwargs: _Response(
            200,
            {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "reasoning_content": "669\n0.90\nclear",
                        }
                    }
                ]
            },
        ),
    )
    assistant = OpenAIVLMAssistant("fake-openai-key")

    assert assistant.ocr_bib(np.ones((10, 10, 3), dtype=np.uint8)) == (
        "669",
        0.90,
        "clear",
    )


def test_openai_compare_persons_matches_detector_pipeline_contract(monkeypatch):
    monkeypatch.setattr(
        vlm_utils.requests,
        "post",
        lambda *args, **kwargs: _Response(200, {"choices": [{"message": {"content": "SAME\n0.84\nhelmet and jersey match"}}]}),
    )
    assistant = OpenAIVLMAssistant("fake-openai-key")

    crops = [np.ones((10, 10, 3), dtype=np.uint8), np.zeros((10, 10, 3), dtype=np.uint8)]
    assert assistant.compare_persons(crops) == ("SAME", 0.84, "helmet and jersey match")
