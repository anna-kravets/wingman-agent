"""Minimal synchronous embedding client for the LLMod.ai endpoint."""

from __future__ import annotations

from dataclasses import dataclass

import httpx


@dataclass(frozen=True)
class LLModEmbeddings:
    api_key: str
    base_url: str
    model: str
    timeout_seconds: float = 45

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            response = httpx.post(
                f"{self.base_url.rstrip('/')}/embeddings",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "input": texts,
                    "encoding_format": "float",
                },
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            items = sorted(payload["data"], key=lambda item: item.get("index", 0))
            vectors = [item["embedding"] for item in items]
        except httpx.HTTPError as exc:
            raise RuntimeError(f"LLMod embedding request failed: {exc}") from exc
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Unexpected LLMod embedding response: {exc}") from exc

        if len(vectors) != len(texts):
            raise RuntimeError(
                "LLMod embedding response count does not match the input count"
            )
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
