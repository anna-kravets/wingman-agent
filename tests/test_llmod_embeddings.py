"""The embedding client uses only the LLMod.ai HTTP configuration."""

import httpx
import pytest

from lib.rag.llmod_embeddings import LLModEmbeddings


def test_embedding_batch_uses_llmod_endpoint_and_restores_index_order(monkeypatch):
    captured = {}

    def ok(url, **kwargs):
        captured.update(url=url, **kwargs)
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [3.0, 4.0]},
                    {"index": 0, "embedding": [1.0, 2.0]},
                ]
            },
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx, "post", ok)
    client = LLModEmbeddings(
        api_key="llmod-test-key",
        base_url="https://example.invalid/v1/",
        model="embedding-model",
    )

    vectors = client.embed_documents(["first", "second"])

    assert vectors == [[1.0, 2.0], [3.0, 4.0]]
    assert captured["url"] == "https://example.invalid/v1/embeddings"
    assert captured["headers"] == {"Authorization": "Bearer llmod-test-key"}
    assert captured["json"] == {
        "model": "embedding-model",
        "input": ["first", "second"],
        "encoding_format": "float",
    }


def test_embedding_response_count_must_match_input(monkeypatch):
    def incomplete(url, **kwargs):
        return httpx.Response(
            200,
            json={"data": [{"index": 0, "embedding": [1.0]}]},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx, "post", incomplete)
    client = LLModEmbeddings("key", "https://example.invalid/v1", "model")

    with pytest.raises(RuntimeError, match="count does not match"):
        client.embed_documents(["first", "second"])
