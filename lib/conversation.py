import os
from datetime import UTC, datetime

from supabase import Client, create_client

_client: Client | None = None


def _is_configured() -> bool:
    """Supabase isn't provisioned yet. Until it is, conversation state is
    best-effort: turns aren't persisted and every call behaves as a single turn.
    Multi-turn switches on by itself once the env vars are set."""
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_KEY")
    return bool(os.environ.get("SUPABASE_URL") and key)


def _get_client() -> Client:
    global _client
    if _client is None:
        url = os.environ["SUPABASE_URL"]
        key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ["SUPABASE_KEY"]
        _client = create_client(url, key)
    return _client


def load_history(owner_id: str, conversation_id: str) -> list[dict]:
    if not _is_configured():
        return []
    res = (
        _get_client()
        .table("conversations")
        .select("history")
        .eq("owner_id", owner_id)
        .eq("conversation_id", conversation_id)
        .execute()
    )
    if not res.data:
        return []
    history = res.data[0].get("history", [])
    return history if isinstance(history, list) else []


def save_history(
    owner_id: str, conversation_id: str, history: list[dict], title: str
) -> None:
    if not _is_configured():
        return
    now = datetime.now(UTC).isoformat()
    _get_client().table("conversations").upsert(
        {
            "owner_id": owner_id,
            "conversation_id": conversation_id,
            "title": title,
            "history": history,
            "updated_at": now,
        },
        on_conflict="owner_id,conversation_id",
    ).execute()


def list_conversations(owner_id: str, limit: int = 30) -> list[dict]:
    if not _is_configured():
        return []
    res = (
        _get_client()
        .table("conversations")
        .select("conversation_id,title,history,created_at,updated_at")
        .eq("owner_id", owner_id)
        .order("updated_at", desc=True)
        .limit(limit)
        .execute()
    )

    conversations = []
    for row in res.data or []:
        history = row.get("history") if isinstance(row, dict) else []
        messages = []
        for turn in history if isinstance(history, list) else []:
            if not isinstance(turn, dict):
                continue
            prompt = turn.get("prompt")
            answer = turn.get("response")
            if isinstance(prompt, str):
                messages.append({"role": "user", "content": prompt})
            if isinstance(answer, str):
                messages.append({"role": "assistant", "content": answer})

        conversations.append(
            {
                "id": row.get("conversation_id"),
                "title": row.get("title") or "New conversation",
                "createdAt": row.get("created_at"),
                "updatedAt": row.get("updated_at"),
                "messages": messages,
            }
        )
    return conversations


def delete_conversation(owner_id: str, conversation_id: str) -> None:
    if not _is_configured():
        return
    (
        _get_client()
        .table("conversations")
        .delete()
        .eq("owner_id", owner_id)
        .eq("conversation_id", conversation_id)
        .execute()
    )
