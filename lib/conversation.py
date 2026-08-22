import logging
import os
from datetime import UTC, datetime

from supabase import Client, create_client

from lib.citations import citations_from_results

logger = logging.getLogger(__name__)

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


def save_steps(
    owner_id: str, conversation_id: str, turn_index: int, steps: list[dict]
) -> None:
    """Persist one turn's execution trace.

    Kept out of `save_history` on purpose. The trace lives in its own table because
    `history` is read in full on the agent path and a trace is far larger than the
    turn it describes; and because a failure here must never cost the passenger their
    conversation, so the caller guards the two writes separately.
    """
    if not _is_configured() or not steps:
        return
    _get_client().table("conversation_traces").upsert(
        {
            "owner_id": owner_id,
            "conversation_id": conversation_id,
            "turn_index": turn_index,
            "steps": steps,
        },
        on_conflict="owner_id,conversation_id,turn_index",
    ).execute()


def _traces(owner_id: str, conversation_ids: list[str]) -> dict[str, dict[int, list]]:
    """Every stored trace for these conversations, as {conversation_id: {turn: steps}}.

    Best-effort by design: the trace is a debugging and demo affordance, so a missing
    table (the migration not run yet) or a failed read must degrade to conversations
    without an "Execution details" panel, not to a 503 on the whole listing.
    """
    if not conversation_ids:
        return {}
    try:
        res = (
            _get_client()
            .table("conversation_traces")
            .select("conversation_id,turn_index,steps")
            .eq("owner_id", owner_id)
            .in_("conversation_id", conversation_ids)
            .execute()
        )
    except Exception:
        logger.exception("Conversation traces could not be loaded; listing without them")
        return {}

    traces: dict[str, dict[int, list]] = {}
    for row in res.data or []:
        steps = row.get("steps")
        if isinstance(steps, list) and steps:
            traces.setdefault(row.get("conversation_id"), {})[row.get("turn_index")] = steps
    return traces


def _messages(history: object, traces: dict[int, list] | None = None) -> list[dict]:
    traces = traces or {}
    messages: list[dict] = []
    # Indexed over the raw list, so the position matches the `turn_index` save_steps
    # stored - a turn skipped below would otherwise shift every later trace by one.
    for index, turn in enumerate(history if isinstance(history, list) else []):
        if not isinstance(turn, dict):
            continue
        prompt = turn.get("prompt")
        answer = turn.get("response")
        if isinstance(prompt, str):
            messages.append({"role": "user", "content": prompt})
        if isinstance(answer, str):
            message = {"role": "assistant", "content": answer}
            citations = citations_from_results(turn.get("results"))
            if citations:
                message["citations"] = citations
            steps = traces.get(index)
            if steps:
                message["steps"] = steps
            messages.append(message)
    return messages


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

    rows = [row for row in res.data or [] if isinstance(row, dict)]
    traces = _traces(owner_id, [row.get("conversation_id") for row in rows
                                if row.get("conversation_id")])

    conversations = []
    for row in rows:
        history = row.get("history")

        conversations.append(
            {
                "id": row.get("conversation_id"),
                "title": row.get("title") or "New conversation",
                "createdAt": row.get("created_at"),
                "updatedAt": row.get("updated_at"),
                "messages": _messages(history, traces.get(row.get("conversation_id"))),
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
    # After the conversation itself, and best-effort: the passenger asked for the
    # conversation to be gone, and it is. An orphaned trace must not turn that into
    # a failure they see.
    try:
        (
            _get_client()
            .table("conversation_traces")
            .delete()
            .eq("owner_id", owner_id)
            .eq("conversation_id", conversation_id)
            .execute()
        )
    except Exception:
        logger.exception("Conversation traces could not be deleted for %s", conversation_id)
