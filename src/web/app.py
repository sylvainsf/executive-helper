"""Web UI for the decision journal — review and rate system decisions."""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.journal import store

router = APIRouter()


class FeedbackRequest(BaseModel):
    functional_rating: str | None = Field(None, pattern=r"^(correct|partial|wrong)$")
    personal_rating: str | None = Field(None, pattern=r"^(helpful|neutral|unhelpful)$")
    notes: str | None = Field(None, max_length=2000)


# ── API endpoints ────────────────────────────────────────────────────────────


@router.get("/api/journal/decisions")
async def list_decisions(
    limit: int = 50,
    offset: int = 0,
    trigger: str | None = None,
    model: str | None = None,
    rating: str | None = None,
):
    return {
        "decisions": store.get_decisions(limit, offset, trigger, model, rating),
        "stats": store.get_stats(),
    }


@router.get("/api/journal/decisions/{decision_id}")
async def get_decision(decision_id: int):
    decision = store.get_decision(decision_id)
    if not decision:
        return {"error": "not found"}, 404
    return decision


@router.post("/api/journal/decisions/{decision_id}/feedback")
async def submit_feedback(decision_id: int, req: FeedbackRequest):
    ok = store.update_feedback(
        decision_id,
        functional_rating=req.functional_rating,
        personal_rating=req.personal_rating,
        notes=req.notes,
    )
    return {"ok": ok}


@router.get("/api/journal/export")
async def export_data(rating: str = "helpful"):
    examples = store.export_for_training(rating_filter=rating)
    return {"count": len(examples), "examples": examples}


@router.get("/api/journal/stats")
async def get_stats():
    return store.get_stats()


# ── HTML UI ──────────────────────────────────────────────────────────────────


@router.get("/journal", response_class=HTMLResponse)
async def journal_page(
    trigger: str | None = None,
    model: str | None = None,
    rating: str | None = None,
):
    decisions = store.get_decisions(limit=100, trigger=trigger, model=model, rating_filter=rating)
    stats = store.get_stats()

    # Inline HTML — avoids Jinja2 template dependency for now
    rows_html = ""
    for d in decisions:
        func_badge = _rating_badge(d.get("functional_rating"), "functional")
        pers_badge = _rating_badge(d.get("personal_rating"), "personal")
        ef_html = ""
        if d.get("ef_intervention"):
            ef_html = f'<div class="ef-intervention"><strong>EF:</strong> {_esc(d["ef_intervention"][:200])}</div>'

        rows_html += f"""
        <div class="decision" id="d-{d['id']}">
            <div class="meta">
                <span class="timestamp">{d['timestamp'][:19]}</span>
                <span class="badge trigger">{_esc(d['trigger'])}</span>
                <span class="badge model">{_esc(d['model'])}</span>
                <span class="badge room">{_esc(d['room'])}</span>
                <span class="badge speaker">{_esc(d['speaker'])}</span>
            </div>
            <div class="input"><strong>Input:</strong> {_esc(d['input_text'][:300])}</div>
            <div class="response"><strong>Response:</strong> {_esc(d['response'][:500])}</div>
            {ef_html}
            <div class="feedback-row">
                <div class="feedback-group">
                    <label>Did it work?</label>
                    {func_badge}
                    <button onclick="rate({d['id']}, 'functional_rating', 'correct')" class="btn-sm">✓</button>
                    <button onclick="rate({d['id']}, 'functional_rating', 'partial')" class="btn-sm">~</button>
                    <button onclick="rate({d['id']}, 'functional_rating', 'wrong')" class="btn-sm">✗</button>
                </div>
                <div class="feedback-group">
                    <label>For me?</label>
                    {pers_badge}
                    <button onclick="rate({d['id']}, 'personal_rating', 'helpful')" class="btn-sm">💚</button>
                    <button onclick="rate({d['id']}, 'personal_rating', 'neutral')" class="btn-sm">💛</button>
                    <button onclick="rate({d['id']}, 'personal_rating', 'unhelpful')" class="btn-sm">🔴</button>
                </div>
                <div class="feedback-group notes-group">
                    <input type="text" id="notes-{d['id']}" placeholder="Notes..."
                           value="{_esc(d.get('notes') or '')}"
                           onkeydown="if(event.key==='Enter')submitNotes({d['id']})">
                    <button onclick="submitNotes({d['id']})" class="btn-sm">Save</button>
                </div>
            </div>
        </div>
        """

    filter_html = ""
    for label, param, values in [
        ("Trigger", "trigger", ["wake_word", "conversation", "routine", "manual"]),
        ("Model", "model", ["ef", "auto"]),
        ("Rating", "rating", ["unrated", "helpful", "neutral", "unhelpful"]),
    ]:
        opts = "".join(f'<a href="?{param}={v}" class="filter-link">{v}</a>' for v in values)
        filter_html += f'<span class="filter-group"><strong>{label}:</strong> {opts}</span>'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Executive Helper — Decision Journal</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #1a1a2e; color: #e0e0e0; padding: 20px; max-width: 900px; margin: 0 auto; }}
  h1 {{ color: #a8d8ea; margin-bottom: 8px; }}
  .stats {{ color: #888; margin-bottom: 20px; font-size: 14px; }}
  .stats span {{ margin-right: 16px; }}
  .filters {{ margin-bottom: 20px; display: flex; gap: 16px; flex-wrap: wrap; }}
  .filter-group {{ font-size: 13px; }}
  .filter-link {{ color: #a8d8ea; margin-left: 6px; text-decoration: none; }}
  .filter-link:hover {{ text-decoration: underline; }}
  .decision {{ background: #16213e; border-radius: 8px; padding: 16px; margin-bottom: 12px;
               border-left: 3px solid #0f3460; }}
  .meta {{ display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 8px; }}
  .timestamp {{ color: #888; font-size: 13px; }}
  .badge {{ font-size: 11px; padding: 2px 8px; border-radius: 10px; }}
  .trigger {{ background: #0f3460; color: #a8d8ea; }}
  .model {{ background: #533483; color: #e0c3fc; }}
  .room {{ background: #1a472a; color: #90ee90; }}
  .speaker {{ background: #4a3728; color: #f0c674; }}
  .input, .response, .ef-intervention {{ font-size: 14px; margin-bottom: 6px; line-height: 1.5; }}
  .ef-intervention {{ background: #1a2a1a; padding: 8px; border-radius: 4px; border-left: 2px solid #4caf50; }}
  .feedback-row {{ display: flex; gap: 16px; margin-top: 10px; align-items: center; flex-wrap: wrap; }}
  .feedback-group {{ display: flex; align-items: center; gap: 4px; font-size: 13px; }}
  .feedback-group label {{ color: #888; margin-right: 4px; }}
  .notes-group {{ flex: 1; }}
  .notes-group input {{ background: #0d1b36; border: 1px solid #333; color: #e0e0e0;
                        padding: 4px 8px; border-radius: 4px; font-size: 13px; width: 100%; min-width: 150px; }}
  .btn-sm {{ background: #0f3460; border: none; color: #e0e0e0; padding: 3px 8px;
             border-radius: 4px; cursor: pointer; font-size: 13px; }}
  .btn-sm:hover {{ background: #1a5276; }}
  .rating-badge {{ font-size: 12px; padding: 1px 6px; border-radius: 6px; }}
  .rating-correct {{ background: #1a472a; color: #4caf50; }}
  .rating-partial {{ background: #4a3728; color: #f0c674; }}
  .rating-wrong {{ background: #4a1a1a; color: #ef5350; }}
  .rating-helpful {{ background: #1a472a; color: #4caf50; }}
  .rating-neutral {{ background: #4a3728; color: #f0c674; }}
  .rating-unhelpful {{ background: #4a1a1a; color: #ef5350; }}
  a.clear {{ color: #888; font-size: 12px; margin-left: 8px; }}
</style>
</head>
<body>
<h1>Decision Journal</h1>
<div class="stats">
  <span>Total: {stats['total']}</span>
  <span>Rated: {stats['rated']}</span>
  <span>💚 {stats['helpful']}</span>
  <span>🔴 {stats['unhelpful']}</span>
  <span>Unrated: {stats['unrated']}</span>
  <a href="/journal" class="clear">clear filters</a>
</div>
<div class="filters">{filter_html}</div>
<div id="decisions">{rows_html}</div>
<script>
async function rate(id, field, value) {{
  const body = {{}};
  body[field] = value;
  await fetch(`/api/journal/decisions/${{id}}/feedback`, {{
    method: 'POST',
    headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify(body)
  }});
  location.reload();
}}
async function submitNotes(id) {{
  const notes = document.getElementById(`notes-${{id}}`).value;
  await fetch(`/api/journal/decisions/${{id}}/feedback`, {{
    method: 'POST',
    headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify({{notes}})
  }});
}}
</script>
</body>
</html>"""
    return HTMLResponse(content=html)


def _esc(s: str) -> str:
    """Basic HTML escaping."""
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _rating_badge(rating: str | None, kind: str) -> str:
    if not rating:
        return '<span class="rating-badge" style="color:#555">—</span>'
    labels = {
        "correct": "✓", "partial": "~", "wrong": "✗",
        "helpful": "💚", "neutral": "💛", "unhelpful": "🔴",
    }
    label = labels.get(rating, rating)
    return f'<span class="rating-badge rating-{rating}">{label}</span>'
