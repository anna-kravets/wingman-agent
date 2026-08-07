import os

from supabase import Client, create_client

_client: Client | None = None


def _get_client() -> Client:
    global _client
    if _client is None:
        url = os.environ["SUPABASE_URL"]
        key = os.environ["SUPABASE_KEY"]
        _client = create_client(url, key)
    return _client


def load_history(conversation_id: str) -> list[dict]:
    res = (
        _get_client()
        .table("conversations")
        .select("history")
        .eq("conversation_id", conversation_id)
        .execute()
    )
    if not res.data:
        return []
    return res.data[0]["history"]


def save_history(conversation_id: str, history: list[dict]) -> None:
    _get_client().table("conversations").upsert(
        {"conversation_id": conversation_id, "history": history}
    ).execute()
