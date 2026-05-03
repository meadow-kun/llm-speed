"""Canned multi-turn agent trace used by the `agent-trace` workload.

A coding-agent transcript: read files, propose a patch, run tests, fix on
failure, run tests again. The fixture is fully synthetic — every line is
written here, no copyrighted material — but it reads like real Python and
a real pytest traceback so prefix-cache hit rates and prefill scaling
behave the way they would on a true coding-agent workload.

Edits to this file change the suite contract: bump SUITE_VERSION.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TraceTurn:
    role: str  # 'user' | 'assistant' | 'tool'
    content: str  # the message body
    expect_output_tokens: int = 0  # for assistant turns: target decode length


SYSTEM_PROMPT = (
    "You are a coding agent. You have three tools:\n"
    "  - read_file(path): returns the file's contents.\n"
    "  - write_file(path, contents): replaces the file at path.\n"
    "  - run_tests(selector): runs the matching pytest selection.\n"
    "Emit a single tool call per turn as <tool_call>...</tool_call>. "
    "Be concise; reason in short bullets. Stop when the user's task is verified by passing tests."
)


# ---------------------------------------------------------------------------
# Synthetic file contents (the "tool" turns)
# ---------------------------------------------------------------------------

_SEARCH_PY_BEFORE = '''\
"""HTTP search endpoint. Currently offset/limit; needs cursor pagination."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from flask import Blueprint, jsonify, request
from werkzeug.exceptions import BadRequest

from ..db.queries import search_documents, count_documents
from ..auth import require_user
from ..tracing import span

LOG = logging.getLogger(__name__)
bp = Blueprint("search", __name__, url_prefix="/search")

DEFAULT_LIMIT = 20
MAX_LIMIT = 100


@dataclass
class SearchParams:
    query: str
    limit: int
    offset: int
    sort: str
    filters: dict

    @classmethod
    def from_request(cls) -> "SearchParams":
        q = request.args.get("q", "").strip()
        if not q:
            raise BadRequest("missing required parameter 'q'")
        try:
            limit = int(request.args.get("limit", DEFAULT_LIMIT))
            offset = int(request.args.get("offset", 0))
        except ValueError as exc:
            raise BadRequest(f"limit/offset must be integers: {exc}") from exc
        if limit < 1 or limit > MAX_LIMIT:
            raise BadRequest(f"limit must be in [1, {MAX_LIMIT}]")
        if offset < 0:
            raise BadRequest("offset must be >= 0")
        sort = request.args.get("sort", "relevance")
        if sort not in {"relevance", "recent", "popular"}:
            raise BadRequest(f"unknown sort: {sort}")
        filters = {}
        for k in ("lang", "type", "author"):
            v = request.args.get(k)
            if v:
                filters[k] = v
        return cls(query=q, limit=limit, offset=offset, sort=sort, filters=filters)


@bp.route("", methods=["GET"])
@require_user
def search(user):
    params = SearchParams.from_request()
    LOG.info("search user=%s q=%r limit=%d offset=%d", user.id, params.query, params.limit, params.offset)
    with span("search.query", user=user.id):
        rows = search_documents(
            query=params.query,
            limit=params.limit,
            offset=params.offset,
            sort=params.sort,
            filters=params.filters,
            user_id=user.id,
        )
    with span("search.count"):
        total = count_documents(query=params.query, filters=params.filters, user_id=user.id)
    return jsonify({
        "items": [_row_to_json(r) for r in rows],
        "total": total,
        "limit": params.limit,
        "offset": params.offset,
        "has_more": params.offset + len(rows) < total,
    })


def _row_to_json(row) -> dict:
    return {
        "id": row.id,
        "title": row.title,
        "snippet": row.snippet,
        "score": float(row.score),
        "created_at": row.created_at.isoformat(),
        "author": row.author,
        "lang": row.lang,
    }


@bp.errorhandler(BadRequest)
def _bad_request(exc):
    return jsonify({"error": str(exc.description)}), 400
'''

_QUERIES_PY_BEFORE = '''\
"""Database query helpers for documents."""

from __future__ import annotations

import logging
from typing import Iterable, Optional

from sqlalchemy import select, func, and_, or_, desc
from sqlalchemy.orm import Session

from .models import Document, DocumentTag
from .session import get_session

LOG = logging.getLogger(__name__)


def _apply_filters(stmt, filters: dict):
    if "lang" in filters:
        stmt = stmt.where(Document.lang == filters["lang"])
    if "type" in filters:
        stmt = stmt.where(Document.kind == filters["type"])
    if "author" in filters:
        stmt = stmt.where(Document.author == filters["author"])
    return stmt


def _apply_sort(stmt, sort: str):
    if sort == "recent":
        return stmt.order_by(desc(Document.created_at))
    if sort == "popular":
        return stmt.order_by(desc(Document.popularity), desc(Document.created_at))
    # relevance: caller has already attached a score column expression
    return stmt.order_by(desc("score"), desc(Document.created_at))


def search_documents(
    *,
    query: str,
    limit: int,
    offset: int,
    sort: str,
    filters: dict,
    user_id: int,
) -> list[Document]:
    with get_session() as session:
        score = func.ts_rank(Document.search_tsv, func.plainto_tsquery("english", query)).label("score")
        stmt = select(Document, score).where(Document.search_tsv.op("@@")(func.plainto_tsquery("english", query)))
        stmt = _apply_filters(stmt, filters)
        stmt = _apply_sort(stmt, sort)
        stmt = stmt.limit(limit).offset(offset)
        rows = session.execute(stmt).all()
        return [r[0] for r in rows]


def count_documents(*, query: str, filters: dict, user_id: int) -> int:
    with get_session() as session:
        stmt = select(func.count(Document.id)).where(
            Document.search_tsv.op("@@")(func.plainto_tsquery("english", query))
        )
        stmt = _apply_filters(stmt, filters)
        return int(session.execute(stmt).scalar_one())
'''

_PATCH_FIRST_ATTEMPT = '''\
Plan:
- Replace `offset` with an opaque `cursor` query param (base64-encoded {created_at, id}).
- Add `next_cursor` to the response when there is a next page.
- Keep `limit` semantics. Drop `total` count from the hot path; expensive at scale.
- Update `search_documents` to accept a cursor instead of offset.

<tool_call>write_file("src/routes/search.py", """
from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass
from typing import Optional

from flask import Blueprint, jsonify, request
from werkzeug.exceptions import BadRequest

from ..db.queries import search_documents
from ..auth import require_user
from ..tracing import span

LOG = logging.getLogger(__name__)
bp = Blueprint('search', __name__, url_prefix='/search')

DEFAULT_LIMIT = 20
MAX_LIMIT = 100


def _encode_cursor(created_at, doc_id: int) -> str:
    payload = json.dumps({'t': created_at.isoformat(), 'id': doc_id}).encode()
    return base64.urlsafe_b64encode(payload).decode().rstrip('=')


def _decode_cursor(raw: str) -> Optional[dict]:
    if not raw:
        return None
    pad = '=' * (-len(raw) % 4)
    try:
        payload = base64.urlsafe_b64decode(raw + pad).decode()
        d = json.loads(payload)
        return {'t': datetime.fromisoformat(d['t']), 'id': int(d['id'])}
    except Exception as exc:
        raise BadRequest(f'invalid cursor: {exc}')


@dataclass
class SearchParams:
    query: str
    limit: int
    cursor: Optional[dict]
    sort: str
    filters: dict


@bp.route('', methods=['GET'])
@require_user
def search(user):
    q = request.args.get('q', '').strip()
    if not q:
        raise BadRequest('missing required parameter q')
    limit = min(int(request.args.get('limit', DEFAULT_LIMIT)), MAX_LIMIT)
    cursor = _decode_cursor(request.args.get('cursor', ''))
    sort = request.args.get('sort', 'relevance')
    filters = {k: v for k in ('lang', 'type', 'author') if (v := request.args.get(k))}
    with span('search.query', user=user.id):
        rows = search_documents(query=q, limit=limit + 1, cursor=cursor, sort=sort, filters=filters, user_id=user.id)
    next_cursor = None
    if len(rows) > limit:
        last = rows[limit - 1]
        next_cursor = _encode_cursor(last.created_at, last.id)
        rows = rows[:limit]
    return jsonify({
        'items': [{'id': r.id, 'title': r.title, 'snippet': r.snippet,
                   'score': float(r.score), 'created_at': r.created_at.isoformat(),
                   'author': r.author, 'lang': r.lang} for r in rows],
        'limit': limit,
        'next_cursor': next_cursor,
    })
""")</tool_call>
'''

_TRACEBACK_MISSING_IMPORT = """\
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-8.2.1, pluggy-1.5.0
rootdir: /workspace/app
plugins: anyio-4.4.0, flask-1.3.0
collected 23 items

tests/routes/test_search.py ....F..F....F..F..F..F.                      [100%]

=================================== FAILURES ===================================
__________________________ test_search_basic_cursor ___________________________

client = <FlaskClient ...>

    def test_search_basic_cursor(client):
        rv = client.get("/search?q=pagination&limit=3")
>       assert rv.status_code == 200, rv.data
E       AssertionError: b'{"error":"Internal Server Error"}'
E       assert 500 == 200

tests/routes/test_search.py:42: AssertionError
------------------------------ Captured log call -------------------------------
ERROR    app.routes.search:search.py:58 search failed: name 'datetime' is not defined
Traceback (most recent call last):
  File "/workspace/app/src/routes/search.py", line 56, in search
    cursor = _decode_cursor(request.args.get('cursor', ''))
  File "/workspace/app/src/routes/search.py", line 28, in _decode_cursor
    return {'t': datetime.fromisoformat(d['t']), 'id': int(d['id'])}
           ^^^^^^^^
NameError: name 'datetime' is not defined
__________________________ test_search_followup_page __________________________
    rv = client.get(f"/search?q=pagination&cursor={cur}")
>   assert rv.status_code == 200
E   assert 500 == 200
E    +  where 500 = <Response [500]>.status_code
__________________________ test_search_empty_query ____________________________
    rv = client.get("/search?q=")
>   assert rv.json["error"] == "missing required parameter q"
E   AssertionError: 500 != 400
__________________________ test_search_bad_cursor _____________________________
    rv = client.get("/search?q=x&cursor=!!notbase64!!")
>   assert rv.status_code == 400
E   assert 500 == 400
__________________________ test_search_filters ________________________________
    rv = client.get("/search?q=cats&lang=en")
E   NameError: name 'datetime' is not defined
__________________________ test_search_sort_recent ____________________________
    rv = client.get("/search?q=cats&sort=recent")
E   NameError: name 'datetime' is not defined
=========================== short test summary info ============================
FAILED tests/routes/test_search.py::test_search_basic_cursor - NameError: name 'datetime' is not defined
FAILED tests/routes/test_search.py::test_search_followup_page
FAILED tests/routes/test_search.py::test_search_empty_query
FAILED tests/routes/test_search.py::test_search_bad_cursor
FAILED tests/routes/test_search.py::test_search_filters - NameError
FAILED tests/routes/test_search.py::test_search_sort_recent - NameError
========================= 6 failed, 17 passed in 1.84s =========================
"""

_PATCH_FIX = '''\
The traceback shows `NameError: name 'datetime' is not defined` inside
`_decode_cursor`. I forgot the import. Adding it and re-issuing the patch.

<tool_call>write_file("src/routes/search.py", """
from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from flask import Blueprint, jsonify, request
from werkzeug.exceptions import BadRequest

from ..db.queries import search_documents
from ..auth import require_user
from ..tracing import span

LOG = logging.getLogger(__name__)
bp = Blueprint('search', __name__, url_prefix='/search')

DEFAULT_LIMIT = 20
MAX_LIMIT = 100


def _encode_cursor(created_at, doc_id: int) -> str:
    payload = json.dumps({'t': created_at.isoformat(), 'id': doc_id}).encode()
    return base64.urlsafe_b64encode(payload).decode().rstrip('=')


def _decode_cursor(raw: str) -> Optional[dict]:
    if not raw:
        return None
    pad = '=' * (-len(raw) % 4)
    try:
        payload = base64.urlsafe_b64decode(raw + pad).decode()
        d = json.loads(payload)
        return {'t': datetime.fromisoformat(d['t']), 'id': int(d['id'])}
    except Exception as exc:
        raise BadRequest(f'invalid cursor: {exc}')


# ... (rest unchanged from the previous patch)
""")</tool_call>
'''

_FINAL_SUMMARY = """\
Done. Changes:
- /search now accepts an opaque `cursor` (base64 of {created_at, id}) instead of `offset`.
- Response carries `next_cursor` when more pages exist; dropped the expensive `total` count.
- `search_documents` now takes `cursor=` and applies a stable (created_at desc, id desc) tiebreak.
- Added a guard so an invalid cursor returns 400, not 500.

The fix from the first failed run was a missing `from datetime import datetime` in
`src/routes/search.py`. All 23 tests in `tests/routes/test_search.py` now pass.
"""


# ---------------------------------------------------------------------------
# The trace
# ---------------------------------------------------------------------------

TRACE: list[TraceTurn] = [
    TraceTurn(
        role="user",
        content=(
            "Add cursor-based pagination to GET /search. Today it uses offset/limit, which "
            "scales badly past ~50k rows and double-counts when new docs land mid-scroll. "
            "Use an opaque base64 cursor encoding (created_at, id), default limit 20, max 100. "
            "Make sure tests/routes/test_search.py still passes. Don't break the public response "
            "shape more than necessary."
        ),
        expect_output_tokens=0,
    ),
    TraceTurn(
        role="assistant",
        content=(
            "I'll start by reading the current /search route, then the DB query layer it calls.\n"
            '<tool_call>read_file("src/routes/search.py")</tool_call>'
        ),
        expect_output_tokens=80,
    ),
    TraceTurn(role="tool", content=_SEARCH_PY_BEFORE, expect_output_tokens=0),
    TraceTurn(
        role="assistant",
        content=(
            "OK — offset/limit shape is straightforward. Now the query helper to see what "
            "API I have for cursor-style queries.\n"
            '<tool_call>read_file("src/db/queries.py")</tool_call>'
        ),
        expect_output_tokens=60,
    ),
    TraceTurn(role="tool", content=_QUERIES_PY_BEFORE, expect_output_tokens=0),
    TraceTurn(role="assistant", content=_PATCH_FIRST_ATTEMPT, expect_output_tokens=300),
    TraceTurn(
        role="tool",
        content="Wrote 84 lines to src/routes/search.py.",
        expect_output_tokens=0,
    ),
    TraceTurn(
        role="assistant",
        content='<tool_call>run_tests("tests/routes/test_search.py")</tool_call>',
        expect_output_tokens=40,
    ),
    TraceTurn(role="tool", content=_TRACEBACK_MISSING_IMPORT, expect_output_tokens=0),
    TraceTurn(role="assistant", content=_PATCH_FIX, expect_output_tokens=200),
    TraceTurn(
        role="tool",
        content="Wrote 84 lines to src/routes/search.py.",
        expect_output_tokens=0,
    ),
    TraceTurn(
        role="assistant",
        content='<tool_call>run_tests("tests/routes/test_search.py")</tool_call>',
        expect_output_tokens=40,
    ),
    TraceTurn(
        role="tool",
        content=(
            "============================= test session starts ==============================\n"
            "platform linux -- Python 3.12.3, pytest-8.2.1, pluggy-1.5.0\n"
            "collected 23 items\n\n"
            "tests/routes/test_search.py .......................                      [100%]\n\n"
            "============================== 23 passed in 1.61s =============================="
        ),
        expect_output_tokens=0,
    ),
    TraceTurn(role="assistant", content=_FINAL_SUMMARY, expect_output_tokens=120),
]


__all__ = ["TraceTurn", "SYSTEM_PROMPT", "TRACE"]
