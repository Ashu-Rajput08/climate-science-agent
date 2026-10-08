"""User-scoped Supabase persistence for conversations and separate experiment artifacts."""
from datetime import datetime, timezone
import uuid


CHAT_TABLE = "climate_chats"
MESSAGE_TABLE = "climate_messages"
EXPERIMENT_TABLE = "climate_experiments"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rows(response) -> list[dict]:
    data = getattr(response, "data", None)
    if data is None:
        return []
    return data if isinstance(data, list) else [data]


def _stored_metadata(metadata: dict | None) -> dict:
    """Persist references and clarification state, never full experiment outputs in chat rows."""
    metadata = metadata or {}
    experiment_id = metadata.get("experiment_id")
    if not experiment_id:
        experiment_id = (metadata.get("analysis", {}).get("experiment", {})
                         .get("experiment_id"))
    stored = {}
    if experiment_id:
        stored["experiment_id"] = experiment_id
    if metadata.get("confirmation"):
        stored["confirmation"] = metadata["confirmation"]
    if metadata.get("research_question") and not experiment_id:
        stored["research_question"] = metadata["research_question"]
    return stored


def _hydrate_experiment(metadata: dict, experiment: dict | None) -> dict:
    if not experiment:
        return metadata
    record = experiment.get("record") or {}
    return {
        **metadata,
        "analysis": {"experiment": record.get("experiment", {}),
                     "results": record.get("results", {})},
        "quality": record.get("data_quality", {}),
        "critic": record.get("critic", {}),
        "report_text": experiment.get("report", ""),
        "report_path": None,
        "record_path": None,
    }


def create_chat(client, user_id: str) -> dict:
    now = _now()
    chat = {"id": uuid.uuid4().hex, "user_id": user_id, "title": "New chat",
            "title_is_custom": False,
            "created_at": now, "updated_at": now, "messages": []}
    client.table(CHAT_TABLE).insert({key: chat[key] for key in
                                     ("id", "user_id", "title", "title_is_custom", "created_at", "updated_at")}).execute()
    return chat


def list_chats(client, user_id: str) -> list[dict]:
    response = (client.table(CHAT_TABLE).select("id,title,title_is_custom,updated_at")
                .eq("user_id", user_id).order("updated_at", desc=True).execute())
    return _rows(response)


def load_chat(chat_id: str, client, user_id: str) -> dict | None:
    chats = _rows(client.table(CHAT_TABLE).select("*").eq("id", chat_id)
                  .eq("user_id", user_id).limit(1).execute())
    if not chats:
        return None
    chat = chats[0]
    messages = _rows(client.table(MESSAGE_TABLE).select("*")
                     .eq("chat_id", chat_id).eq("user_id", user_id)
                     .order("created_at").execute())
    experiment_ids = sorted({(message.get("metadata") or {}).get("experiment_id")
                             for message in messages
                             if (message.get("metadata") or {}).get("experiment_id")})
    experiments = {}
    if experiment_ids:
        rows = _rows(client.table(EXPERIMENT_TABLE).select("id,record,report")
                     .eq("user_id", user_id).in_("id", experiment_ids).execute())
        experiments = {row["id"]: row for row in rows}
    for message in messages:
        message["metadata"] = _hydrate_experiment(
            message.get("metadata") or {},
            experiments.get((message.get("metadata") or {}).get("experiment_id")),
        )
    chat["messages"] = messages
    return chat


def save_chat(chat: dict, client, user_id: str) -> None:
    """Persist chat metadata and messages while stripping attached experiment payloads."""
    now = _now()
    chat["updated_at"] = now
    client.table(CHAT_TABLE).update({
        "title": chat.get("title", "New chat"), "updated_at": now,
    }).eq("id", chat["id"]).eq("user_id", user_id).execute()
    rows = []
    for message in chat.get("messages", []):
        message.setdefault("id", uuid.uuid4().hex)
        rows.append({
            "id": message["id"], "chat_id": chat["id"], "user_id": user_id,
            "role": message["role"], "content": message["content"],
            "metadata": _stored_metadata(message.get("metadata")),
            "created_at": message.get("created_at") or now,
        })
    if rows:
        client.table(MESSAGE_TABLE).upsert(rows, on_conflict="id").execute()


def add_message(chat: dict, role: str, content: str, client, user_id: str,
                metadata: dict | None = None) -> dict:
    if role not in {"user", "assistant"}:
        raise ValueError("Chat message role must be user or assistant")
    now = _now()
    message = {"id": uuid.uuid4().hex, "role": role, "content": content,
               "metadata": metadata or {}, "created_at": now}
    client.table(MESSAGE_TABLE).insert({
        "id": message["id"], "chat_id": chat["id"], "user_id": user_id,
        "role": role, "content": content,
        "metadata": _stored_metadata(metadata), "created_at": now,
    }).execute()
    chat["messages"].append(message)
    chat["updated_at"] = now
    client.table(CHAT_TABLE).update({"updated_at": now}).eq("id", chat["id"]).eq("user_id", user_id).execute()
    return chat


def rename_chat(client, user_id: str, chat_id: str, title: str) -> str:
    """Persist a user-chosen conversation name, scoped to its owner."""
    clean_title = " ".join((title or "").split())
    if not clean_title:
        raise ValueError("Enter a chat name before saving.")
    if len(clean_title) > 64:
        raise ValueError("Chat names must be 64 characters or fewer.")
    (client.table(CHAT_TABLE).update({"title": clean_title, "title_is_custom": True, "updated_at": _now()})
     .eq("id", chat_id).eq("user_id", user_id).execute())
    return clean_title


def save_experiment(client, user_id: str, chat_id: str, record: dict) -> str:
    """Store the report and structured run separately from the chat message history."""
    experiment = record.get("experiment", {})
    experiment_id = experiment.get("experiment_id")
    if not experiment_id:
        raise ValueError("Experiment record is missing an experiment_id")
    persisted_record = {
        "experiment": experiment,
        "data_quality": record.get("data_quality", {}),
        "results": record.get("results", {}),
        "critic": record.get("critic", {}),
        "experiment_history": record.get("experiment_history", []),
    }
    client.table(EXPERIMENT_TABLE).upsert({
        "id": experiment_id, "chat_id": chat_id, "user_id": user_id,
        "record": persisted_record, "report": record.get("report", ""),
    }, on_conflict="id").execute()
    return experiment_id
