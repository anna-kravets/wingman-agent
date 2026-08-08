import os

from supabase import Client, create_client

_client: Client | None = None


def _is_configured() -> bool:
    """Supabase isn't provisioned yet. Until it is, conversation state is
    best-effort: turns aren't persisted and every call behaves as a single turn.
    Multi-turn switches on by itself once the env vars are set."""
    return bool(os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_KEY"))


def _get_client() -> Client:
    global _client
    if _client is None:
        url = os.environ["SUPABASE_URL"]
        key = os.environ["SUPABASE_KEY"]
        _client = create_client(url, key)
    return _client


def load_history(conversation_id: str) -> list[dict]:
    if not _is_configured():
        return []
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
    if not _is_configured():
        return
    _get_client().table("conversations").upsert(
        {"conversation_id": conversation_id, "history": history}
    ).execute()
