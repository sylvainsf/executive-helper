"""Decision journal — SQLite store for logging and reviewing system decisions."""

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

_DB_PATH = Path("data/journal.db")
_conn: sqlite3.Connection | None = None


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA journal_mode=WAL")
        _init_schema(_conn)
    return _conn


def _init_schema(conn: sqlite3.Connection):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            room TEXT NOT NULL DEFAULT 'unknown',
            speaker TEXT NOT NULL DEFAULT 'unknown',
            trigger TEXT NOT NULL DEFAULT 'unknown',
            input_text TEXT NOT NULL,
            model TEXT NOT NULL,
            response TEXT NOT NULL,
            ef_intervention TEXT,
            functional_rating TEXT,
            personal_rating TEXT,
            notes TEXT,
            exported INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS pending_reminders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            fire_at TEXT NOT NULL,
            room TEXT NOT NULL DEFAULT 'unknown',
            label TEXT NOT NULL,
            action_type TEXT NOT NULL DEFAULT 'reminder',
            conversation_context TEXT,
            original_user_input TEXT,
            timer_entity_id TEXT,
            status TEXT NOT NULL DEFAULT 'pending'
        );

        CREATE INDEX IF NOT EXISTS idx_decisions_timestamp ON decisions(timestamp);
        CREATE INDEX IF NOT EXISTS idx_decisions_model ON decisions(model);
        CREATE INDEX IF NOT EXISTS idx_decisions_trigger ON decisions(trigger);
        CREATE INDEX IF NOT EXISTS idx_reminders_status ON pending_reminders(status);
    """)


async def log_decision(
    room: str = "unknown",
    speaker: str = "unknown",
    input_text: str = "",
    model: str = "unknown",
    response: str = "",
    ef_intervention: str | None = None,
    trigger: str = "unknown",
) -> int:
    """Log a system decision to the journal. Returns the decision ID."""
    conn = _get_conn()
    cursor = conn.execute(
        """INSERT INTO decisions (timestamp, room, speaker, trigger, input_text, model, response, ef_intervention)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            datetime.now(timezone.utc).isoformat(),
            room,
            speaker,
            trigger,
            input_text,
            model,
            response,
            ef_intervention,
        ),
    )
    conn.commit()
    return cursor.lastrowid


def get_decisions(
    limit: int = 50,
    offset: int = 0,
    trigger: str | None = None,
    model: str | None = None,
    rating_filter: str | None = None,
) -> list[dict]:
    """Retrieve decisions from the journal."""
    conn = _get_conn()
    query = "SELECT * FROM decisions WHERE 1=1"
    params: list = []

    if trigger:
        query += " AND trigger = ?"
        params.append(trigger)
    if model:
        query += " AND model = ?"
        params.append(model)
    if rating_filter == "unrated":
        query += " AND functional_rating IS NULL AND personal_rating IS NULL"
    elif rating_filter == "helpful":
        query += " AND personal_rating = 'helpful'"
    elif rating_filter == "unhelpful":
        query += " AND personal_rating = 'unhelpful'"

    query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    rows = conn.execute(query, params).fetchall()
    return [dict(row) for row in rows]


def get_decision(decision_id: int) -> dict | None:
    """Get a single decision by ID."""
    conn = _get_conn()
    row = conn.execute("SELECT * FROM decisions WHERE id = ?", (decision_id,)).fetchone()
    return dict(row) if row else None


def update_feedback(
    decision_id: int,
    functional_rating: str | None = None,
    personal_rating: str | None = None,
    notes: str | None = None,
) -> bool:
    """Update feedback on a decision."""
    conn = _get_conn()
    fields = []
    params: list = []

    if functional_rating is not None:
        fields.append("functional_rating = ?")
        params.append(functional_rating)
    if personal_rating is not None:
        fields.append("personal_rating = ?")
        params.append(personal_rating)
    if notes is not None:
        fields.append("notes = ?")
        params.append(notes)

    if not fields:
        return False

    params.append(decision_id)
    conn.execute(f"UPDATE decisions SET {', '.join(fields)} WHERE id = ?", params)
    conn.commit()
    return True


def export_for_training(
    rating_filter: str | None = "helpful",
    format: str = "jsonl",
) -> list[dict]:
    """Export rated decisions as training data.

    Args:
        rating_filter: "helpful", "unhelpful", or None for all rated.
        format: "jsonl" (chat format for fine-tuning).

    Returns:
        List of training examples in chat format.
    """
    conn = _get_conn()
    query = "SELECT * FROM decisions WHERE personal_rating IS NOT NULL"
    params: list = []

    if rating_filter:
        query += " AND personal_rating = ?"
        params.append(rating_filter)

    query += " AND exported = 0 ORDER BY timestamp"
    rows = conn.execute(query, params).fetchall()

    examples = []
    for row in rows:
        row = dict(row)
        example = {
            "messages": [
                {"role": "user", "content": row["input_text"]},
                {"role": "assistant", "content": row["response"]},
            ],
            "model": row["model"],
            "rating": row["personal_rating"],
            "functional_rating": row["functional_rating"],
        }
        if row["ef_intervention"]:
            example["ef_intervention"] = row["ef_intervention"]
        examples.append(example)

    # Mark as exported
    ids = [dict(r)["id"] for r in rows]
    if ids:
        placeholders = ",".join("?" * len(ids))
        conn.execute(f"UPDATE decisions SET exported = 1 WHERE id IN ({placeholders})", ids)
        conn.commit()

    return examples


def get_stats() -> dict:
    """Get journal statistics."""
    conn = _get_conn()
    total = conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
    rated = conn.execute(
        "SELECT COUNT(*) FROM decisions WHERE personal_rating IS NOT NULL"
    ).fetchone()[0]
    helpful = conn.execute(
        "SELECT COUNT(*) FROM decisions WHERE personal_rating = 'helpful'"
    ).fetchone()[0]
    unhelpful = conn.execute(
        "SELECT COUNT(*) FROM decisions WHERE personal_rating = 'unhelpful'"
    ).fetchone()[0]
    by_model = dict(
        conn.execute(
            "SELECT model, COUNT(*) FROM decisions GROUP BY model"
        ).fetchall()
    )
    by_trigger = dict(
        conn.execute(
            "SELECT trigger, COUNT(*) FROM decisions GROUP BY trigger"
        ).fetchall()
    )

    return {
        "total": total,
        "rated": rated,
        "unrated": total - rated,
        "helpful": helpful,
        "unhelpful": unhelpful,
        "neutral": rated - helpful - unhelpful,
        "by_model": by_model,
        "by_trigger": by_trigger,
    }


# ── Pending Reminders ────────────────────────────────────────────────────────


async def create_reminder(
    room: str,
    label: str,
    minutes: int,
    action_type: str = "reminder",
    conversation_context: str = "",
    original_user_input: str = "",
    timer_entity_id: str = "",
) -> int:
    """Store a pending reminder with its conversation context.

    When the HA timer fires, the callback endpoint looks up this reminder
    to reconstruct context for the EF model.
    """
    conn = _get_conn()
    now = datetime.now(timezone.utc)
    fire_at = now + timedelta(minutes=minutes)
    cursor = conn.execute(
        """INSERT INTO pending_reminders
           (created_at, fire_at, room, label, action_type,
            conversation_context, original_user_input, timer_entity_id, status)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending')""",
        (
            now.isoformat(),
            fire_at.isoformat(),
            room,
            label,
            action_type,
            conversation_context,
            original_user_input,
            timer_entity_id,
        ),
    )
    conn.commit()
    return cursor.lastrowid


def get_pending_reminder(reminder_id: int) -> dict | None:
    """Retrieve a pending reminder by ID."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM pending_reminders WHERE id = ? AND status = 'pending'",
        (reminder_id,),
    ).fetchone()
    return dict(row) if row else None


def complete_reminder(reminder_id: int):
    """Mark a reminder as completed (fired)."""
    conn = _get_conn()
    conn.execute(
        "UPDATE pending_reminders SET status = 'completed' WHERE id = ?",
        (reminder_id,),
    )
    conn.commit()


def get_pending_reminder_by_timer(timer_entity_id: str) -> dict | None:
    """Look up a pending reminder by its HA timer entity ID."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM pending_reminders WHERE timer_entity_id = ? AND status = 'pending'",
        (timer_entity_id,),
    ).fetchone()
    return dict(row) if row else None
