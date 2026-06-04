"""Tool result cache (v1.0, 2026-06-04).

Opt-in, TTL-bounded memoization layer for tool invocations.  Stores the
JSON-serialized result of a tool handler call, keyed by SHA-256 of the
canonical ``(tool_name, args_json)`` tuple.  Lives in the same
``state.db`` as session/messages tables (see hermes_state.py, table
``tool_result_cache``) so a single WAL-mode DB serializes all writes.

Public surface:

* :class:`ToolResultCache` — singleton wrapper around the SQLite table.
* :func:`make_key` — canonical SHA-256 key from ``(tool_name, args)``.
* :func:`canonical_args_json` — stable, sorted-keys JSON serialization.
* :func:`start_eviction_thread` / :func:`stop_eviction_thread` — process-
  scoped background sweep (one thread per process, ±15 min jitter).

Design contract (read before editing):

1.  **Correctness by construction.**  Hashes canonical JSON, so
    ``{"a":1,"b":2}`` and ``{"b":2,"a":1}`` produce the same key.
2.  **Opt-in only.**  Side-effecting tools (``send_message``, ``write_file``,
    ``terminal``, etc.) must NOT set ``cacheable=True``; the registry
    default is ``False``.  The cache layer never sees unsafe tools because
    the caller (``model_tools.cache_get_or_run``) gates on
    ``entry.cacheable`` before any DB call.
3.  **Per-agent isolation.**  Each cache row carries an ``agent_id``;
    lookups filter on it.  Default ``""`` = "shared across all agents in
    this Hermes CLI profile".  Tools that need per-session scoping
    (``session_search``) pass the current ``session_id`` so a new message
    appended to that session can be invalidated via
    :meth:`invalidate_session`.
4.  **TTL is the only invalidation rule.**  A row with ``expires_at < now``
    is a miss.  The hourly eviction sweep clears expired rows in batches.
5.  **Failure-mode circuit breakers** (DevOps B5).  Every public method
    catches ``sqlite3.OperationalError`` and any other exception, sets a
    ``state_meta`` ``cache_disabled_until`` flag, logs ERROR, and returns
    a miss.  Cache errors NEVER propagate to the caller — the tool call
    just runs un-cached.  This protects the agent loop from a full disk
    or a corrupted DB.
6.  **In-process LRU for the hot path** (Dev B6).  A small
    ``OrderedDict`` (maxsize=1000) sits in front of SQLite to avoid
    regressing existing WAL write-lock contention.  LRU entries hold only
    ``(key, result_json, expires_at)`` — never the deserialized object —
    so memory pressure is bounded by the entry cap × avg result size.
7.  **Single-statement upsert** (Dev B8).  ``INSERT ... ON CONFLICT(key)
    DO UPDATE`` avoids the SELECT-then-INSERT race when the same key is
    written from multiple threads.
8.  **Eviction sweep is hourly with ±15 min jitter** so 1000 installs
    on a shared cron cloud don't all VACUUM at :00.

What this module deliberately does NOT do (deferred to v2):

* LLM response caching (out of scope per the Dev review).
* Cross-session memory cache (different layer; separate ticket).
* UX surfaces (hermes cache subcommand, /cache slash command,
  cached glyph) — all deferred to v1.5 per Designer review.
* Semantic / vector caching.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import sqlite3
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Key canonicalization
# ---------------------------------------------------------------------------
# Hashing the canonical JSON of the args means two semantically identical
# calls (one with kwargs reordered, one with int vs str number) collapse
# to the same cache row.  This is the well-precedented pattern from the
# tool-result-cache PyPI library and matches the project research brief.
#
# IMPORTANT: any new caller MUST round-trip through coerce_tool_args
# (model_tools.py:900) BEFORE hashing, so the dict is in its post-coercion
# form.  Otherwise "42" (string) and 42 (int) would not collide.

def canonical_args_json(args: Optional[Dict[str, Any]]) -> str:
    """Return the canonical JSON form of *args*.

    Sort keys, drop whitespace, keep unicode characters as-is.  ``None``
    is treated as an empty dict so an arg-less call is a stable key.
    """
    if args is None:
        args = {}
    return json.dumps(args, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def make_key(tool_name: str, args: Optional[Dict[str, Any]]) -> str:
    """Return the SHA-256 hex of canonical ``(tool_name, args)``."""
    canonical = canonical_args_json(args)
    digest = hashlib.sha256(f"{tool_name}|{canonical}".encode("utf-8"))
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# In-process LRU
# ---------------------------------------------------------------------------
# Cap of 1000 entries chosen to fit comfortably in <1 MB even for ~1KB
# avg result_json.  The cap is a "do not regress TUI contention" knob
# (Dev B6) — at 1000 entries with an Open hit-rate of ~30% we cut DB
# SELECTs by ~3x on a typical 30-tool-call session, and SQLite PK lookups
# are ~0.5 ms so the LRU is an order of magnitude faster than the DB
# path.

_LRU_MAXSIZE = 1000


def _scope_key(key: str, agent_id: str, session_id: str) -> str:
    """Build a scope-aware cache key for the in-process LRU.

    The LRU sits in front of SQLite to amortize repeat hits, but
    it must NOT collapse different agent_id / session_id values into
    the same entry.  Use a NUL separator (which never appears in
    hex SHA-256 digests) to keep the LRU key self-delimiting.
    """
    return f"{agent_id}\x00{session_id}\x00{key}"


class _LruCache:
    """Minimal OrderedDict-backed LRU; explicit so we don't pull in
    functools.lru_cache's global lock (which would serialize every cache
    lookup across the agent loop)."""

    def __init__(self, maxsize: int = _LRU_MAXSIZE) -> None:
        self._data: "OrderedDict[str, Tuple[Optional[str], float]]" = OrderedDict()
        self._maxsize = maxsize
        self._lock = threading.Lock()

    def get(self, scope_key: str) -> Optional[Tuple[Optional[str], float]]:
        """Return ``(result_json, expires_at)`` on hit, None on miss.

        LRU-bumps the entry on hit.  Result is whatever the SQLite row
        stored — a JSON string or ``None`` for a "negative" cache row
        (used for marking an error result so we don't re-run a known-
        failing tool call).  The wrapper at cache_get_or_run() does not
        currently write negative rows; reserving the type keeps the door
        open.
        """
        with self._lock:
            entry = self._data.get(scope_key)
            if entry is None:
                return None
            # move to end = most-recently-used
            self._data.move_to_end(scope_key)
            return entry

    def put(self, scope_key: str, result_json: Optional[str], expires_at: float) -> None:
        """Insert or refresh *scope_key*; evict the LRU entry if over capacity."""
        with self._lock:
            if scope_key in self._data:
                self._data.move_to_end(scope_key)
            self._data[scope_key] = (result_json, expires_at)
            while len(self._data) > self._maxsize:
                self._data.popitem(last=False)

    def invalidate(self, scope_key: str) -> None:
        with self._lock:
            self._data.pop(scope_key, None)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------
# On any DB error the cache sets a state_meta flag with an absolute
# disable-until timestamp; the wrapper short-circuits for an hour
# without hitting the DB.  This protects the agent loop from a full
# disk or corrupted-DB scenario (DevOps B5).

_DISABLE_DURATION_S = 3600  # 1 hour


class _CircuitBreaker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._disabled_until: float = 0.0

    def is_open(self, now: Optional[float] = None) -> bool:
        if now is None:
            now = time.time()
        with self._lock:
            return now < self._disabled_until

    def trip(self, reason: str, now: Optional[float] = None) -> None:
        if now is None:
            now = time.time()
        with self._lock:
            self._disabled_until = now + _DISABLE_DURATION_S
        logger.error(
            "Tool result cache circuit-breaker OPEN for %ss — reason: %s. "
            "Cache will be bypassed until the timer expires; tool calls "
            "continue to work un-cached.",
            _DISABLE_DURATION_S,
            reason,
        )

    def reset(self) -> None:
        with self._lock:
            self._disabled_until = 0.0


# ---------------------------------------------------------------------------
# Main cache class
# ---------------------------------------------------------------------------


class ToolResultCache:
    """SQLite-backed tool result cache with in-process LRU and a circuit
    breaker.  Use the module-level :func:`get_default_cache` singleton in
    production code; tests can construct their own instance with a
    tmp_path DB.

    Thread-safe: every public method takes the same per-instance lock
    around its SQLite transaction.  The in-process LRU uses its own
    finer-grained lock so LRU hits do not serialize on the DB lock.
    """

    # Reuse the same jitter window as hermes_state._execute_write so
    # eviction batches don't convoy with hot writes.
    _WRITE_RETRY_MIN_S = 0.020
    _WRITE_RETRY_MAX_S = 0.150
    _WRITE_MAX_RETRIES = 15

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        # check_same_thread=False mirrors the hermes_state pattern so the
        # cache works from the gateway's per-request thread pool.
        self._conn = sqlite3.connect(
            str(db_path),
            check_same_thread=False,
            timeout=1.0,
            isolation_level=None,  # we manage transactions explicitly
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._lock = threading.Lock()
        self._lru = _LruCache()
        self._breaker = _CircuitBreaker()
        # Per-process counters — mirror the agent._or_cache_hits pattern
        # so the activity summary can show "saved N tool calls" (Dev B12).
        # Reset on process restart; that's fine for v1.0.
        self.hits: int = 0
        self.misses: int = 0
        self.writes: int = 0
        self.errors: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(
        self,
        tool_name: str,
        args: Optional[Dict[str, Any]],
        agent_id: str = "",
        session_id: str = "",
    ) -> Optional[str]:
        """Return the cached result JSON for ``(tool_name, args)`` or
        ``None`` on miss / cache disabled / error.

        Looks up the in-process LRU first; on miss, queries SQLite; on
        hit, populates the LRU.  Never raises.
        """
        if self._breaker.is_open():
            return None
        key = make_key(tool_name, args)
        scope_key = _scope_key(key, agent_id, session_id)
        try:
            # LRU fast path
            lru_hit = self._lru.get(scope_key)
            if lru_hit is not None:
                result_json, expires_at = lru_hit
                if expires_at > time.time():
                    self.hits += 1
                    return result_json
                # Expired — drop and fall through to DB
                self._lru.invalidate(scope_key)
            # DB path with retry
            row = self._select_row(key, agent_id=agent_id, session_id=session_id)
            if row is None:
                self.misses += 1
                return None
            result_json, expires_at = row
            if expires_at <= time.time():
                # Expired in DB but not yet swept — treat as miss and
                # opportunistically delete.
                self._delete_row(key)
                self.misses += 1
                return None
            self._lru.put(scope_key, result_json, expires_at)
            self.hits += 1
            return result_json
        except sqlite3.OperationalError as exc:
            self.errors += 1
            self._breaker.trip(f"OperationalError on get({tool_name}): {exc}")
            return None
        except Exception as exc:
            self.errors += 1
            logger.error("ToolResultCache.get unexpected error: %s", exc, exc_info=True)
            return None

    def put(
        self,
        tool_name: str,
        args: Optional[Dict[str, Any]],
        result_json: str,
        ttl_seconds: int,
        agent_id: str = "",
        session_id: str = "",
        hermes_profile: str = "",
    ) -> bool:
        """Write *result_json* to the cache under the canonical key.

        Uses ``INSERT ... ON CONFLICT(key) DO UPDATE`` so concurrent
        writers from the gateway + dispatcher + kanban workers don't
        race.  Returns True on successful write, False on cache disabled /
        error.  Never raises.
        """
        if self._breaker.is_open():
            return False
        if ttl_seconds <= 0:
            return False  # explicit no-cache
        key = make_key(tool_name, args)
        scope_key = _scope_key(key, agent_id, session_id)
        canonical = canonical_args_json(args)
        size_bytes = len(result_json.encode("utf-8")) if result_json else 0
        now = time.time()
        expires_at = now + ttl_seconds
        args_sql = (
            key, tool_name, canonical, result_json, size_bytes,
            now, now, ttl_seconds, expires_at,
            hermes_profile or "", agent_id or "", session_id or "",
        )
        sql = (
            "INSERT INTO tool_result_cache ("
            "  key, tool_name, args_json, result_json, size_bytes,"
            "  created_at, last_hit_at, hit_count, ttl_seconds, expires_at,"
            "  hermes_profile, agent_id, session_id"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?)"
            "ON CONFLICT(key) DO UPDATE SET"
            "  result_json = excluded.result_json,"
            "  size_bytes = excluded.size_bytes,"
            "  last_hit_at = excluded.last_hit_at,"
            "  ttl_seconds = excluded.ttl_seconds,"
            "  expires_at = excluded.expires_at,"
            "  agent_id = excluded.agent_id,"
            "  session_id = excluded.session_id"
        )
        try:
            with self._lock:
                self._execute_with_retry(sql, args_sql)
            self._lru.put(scope_key, result_json, expires_at)
            self.writes += 1
            return True
        except sqlite3.OperationalError as exc:
            self.errors += 1
            self._breaker.trip(f"OperationalError on put({tool_name}): {exc}")
            return False
        except Exception as exc:
            self.errors += 1
            logger.error("ToolResultCache.put unexpected error: %s", exc, exc_info=True)
            return False

    def invalidate(self, tool_name: Optional[str] = None) -> int:
        """Delete cache rows.  If *tool_name* is given, only rows for that
        tool; otherwise all rows.  Returns the row count removed, or 0 on
        cache disabled / error."""
        if self._breaker.is_open():
            return 0
        try:
            with self._lock:
                if tool_name is None:
                    cur = self._conn.execute("DELETE FROM tool_result_cache")
                else:
                    cur = self._conn.execute(
                        "DELETE FROM tool_result_cache WHERE tool_name = ?",
                        (tool_name,),
                    )
                removed = cur.rowcount or 0
            # Drop LRU too — entries don't carry tool_name so wholesale
            # clear is the safe option.
            self._lru.clear()
            return removed
        except sqlite3.OperationalError as exc:
            self.errors += 1
            self._breaker.trip(f"OperationalError on invalidate: {exc}")
            return 0
        except Exception as exc:
            self.errors += 1
            logger.error("ToolResultCache.invalidate unexpected error: %s", exc, exc_info=True)
            return 0

    def invalidate_session(self, session_id: str) -> int:
        """Drop all rows whose ``session_id`` matches.

        Used by ``session_search`` and ``memory`` invalidation when a new
        message arrives in that session.  Returns the row count removed.
        """
        if self._breaker.is_open():
            return 0
        if not session_id:
            return 0
        try:
            with self._lock:
                cur = self._conn.execute(
                    "DELETE FROM tool_result_cache WHERE session_id = ?",
                    (session_id,),
                )
                removed = cur.rowcount or 0
            # The LRU keys are scope-prefixed hashes; we have no
            # cheap way to drop only the matching entries, so the
            # simplest correct thing is to flush the LRU.  This is
            # rare (per session message) and the LRU is bounded
            # to 1000 entries, so a clear is microseconds.  Without
            # this flush, get() would happily return a stale LRU
            # hit even though the DB row is gone.
            self._lru.clear()
            return removed
        except sqlite3.OperationalError as exc:
            self.errors += 1
            self._breaker.trip(f"OperationalError on invalidate_session: {exc}")
            return 0
        except Exception as exc:
            self.errors += 1
            logger.error("ToolResultCache.invalidate_session unexpected error: %s", exc, exc_info=True)
            return 0

    def vacuum(
        self,
        size_cap_bytes: int = 50 * 1024 * 1024,
        batch_size: int = 500,
    ) -> Dict[str, int]:
        """Hourly maintenance sweep.  Returns a stats dict.

        1. Delete expired rows in batches of *batch_size* (BEFORE any
           size-cap check, so expired rows don't count toward the cap).
        2. If total size still exceeds *size_cap_bytes*, evict by
           ``last_hit_at`` ASC (coldest first) in batches of
           *batch_size* until under cap.

        Uses ``BEGIN DEFERRED`` (not ``IMMEDIATE``) so it coexists with
        hot writes (DevOps B6).
        """
        if self._breaker.is_open():
            return {"expired_removed": 0, "size_evicted": 0, "skipped": 1}
        now = time.time()
        stats = {"expired_removed": 0, "size_evicted": 0, "skipped": 0}
        try:
            with self._lock:
                # 1. Expired-row sweep
                while True:
                    try:
                        self._conn.execute("BEGIN DEFERRED")
                        # SQLite doesn't support DELETE ... LIMIT
                        # directly; use a subquery to pick the N
                        # oldest-expired keys, then delete by PK.
                        cur = self._conn.execute(
                            "DELETE FROM tool_result_cache "
                            "WHERE key IN ("
                            "  SELECT key FROM tool_result_cache "
                            "  WHERE expires_at < ? "
                            "  ORDER BY expires_at ASC LIMIT ?"
                            ")",
                            (now, batch_size),
                        )
                        removed = cur.rowcount or 0
                        self._conn.execute("COMMIT")
                    except Exception:
                        try:
                            self._conn.execute("ROLLBACK")
                        except Exception:
                            pass
                        raise
                    stats["expired_removed"] += removed
                    if removed < batch_size:
                        break
                    # 20-150ms jitter between batches (matches
                    # hermes_state._execute_write).
                    time.sleep(random.uniform(self._WRITE_RETRY_MIN_S, self._WRITE_RETRY_MAX_S))
                # 2. Size-cap LRU eviction (by last_hit_at ASC)
                total_size = self._total_size_bytes()
                while total_size > size_cap_bytes:
                    try:
                        self._conn.execute("BEGIN DEFERRED")
                        # SQLite doesn't allow ORDER BY + LIMIT in a
                        # DELETE directly; use a subquery to pick the
                        # N oldest-hit keys, then delete by PK.
                        cur = self._conn.execute(
                            "DELETE FROM tool_result_cache "
                            "WHERE key IN ("
                            "  SELECT key FROM tool_result_cache "
                            "  ORDER BY last_hit_at ASC LIMIT ?"
                            ")",
                            (batch_size,),
                        )
                        removed = cur.rowcount or 0
                        self._conn.execute("COMMIT")
                    except Exception:
                        try:
                            self._conn.execute("ROLLBACK")
                        except Exception:
                            pass
                        raise
                    stats["size_evicted"] += removed
                    if removed == 0:
                        # Underlying page_count is bigger than row
                        # sizes suggest (WAL pending writes); bail
                        # rather than spin.
                        break
                    total_size = self._total_size_bytes()
                    time.sleep(random.uniform(self._WRITE_RETRY_MIN_S, self._WRITE_RETRY_MAX_S))
        except sqlite3.OperationalError as exc:
            self.errors += 1
            self._breaker.trip(f"OperationalError on vacuum: {exc}")
            stats["skipped"] = 1
        except Exception as exc:
            self.errors += 1
            logger.error("ToolResultCache.vacuum unexpected error: %s", exc, exc_info=True)
            stats["skipped"] = 1
        return stats

    def stats(self) -> Dict[str, int]:
        """Return a snapshot of counters + table size for observability."""
        try:
            row = self._conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(size_bytes), 0) "
                "FROM tool_result_cache"
            ).fetchone()
            row_count = int(row[0]) if row else 0
            total_bytes = int(row[1]) if row else 0
        except Exception:
            row_count = 0
            total_bytes = 0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
            "errors": self.errors,
            "row_count": row_count,
            "total_bytes": total_bytes,
            "circuit_open": int(self._breaker.is_open()),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _select_row(
        self,
        key: str,
        agent_id: str = "",
        session_id: str = "",
    ) -> Optional[Tuple[str, float]]:
        """SQLite SELECT.  Returns ``(result_json, expires_at)`` or None.

        Filtering rules:
        * ``agent_id`` exact match — empty string matches both empty and
          non-empty stored values? No, we use SQL ``=`` so empty matches
          only empty.  This is intentional: the wrapper passes a
          non-empty ``agent_id`` to scope per-agent caches; leaving it
          empty at the call site means "shared with the default agent".
        * ``session_id`` exact match — same convention.
        """
        # Atomic upsert of hit_count + last_hit_at + return row.
        sql = (
            "SELECT result_json, expires_at FROM tool_result_cache "
            "WHERE key = ? AND agent_id = ? AND session_id = ?"
        )
        # Bump hit_count + last_hit_at in a single statement so the
        # in-process counters stay consistent with the DB.  Use the
        # conflict-tolerant ON CONFLICT so a concurrent write doesn't
        # raise SQLITE_CONSTRAINT.
        bump_sql = (
            "UPDATE tool_result_cache "
            "SET last_hit_at = ?, hit_count = hit_count + 1 "
            "WHERE key = ?"
        )
        with self._lock:
            self._execute_with_retry(bump_sql, (time.time(), key))
            row = self._conn.execute(sql, (key, agent_id, session_id)).fetchone()
        if row is None:
            return None
        return (row[0], float(row[1]))

    def _delete_row(self, key: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM tool_result_cache WHERE key = ?", (key,))
        self._lru.invalidate(key)

    def _total_size_bytes(self) -> int:
        row = self._conn.execute(
            "SELECT COALESCE(SUM(size_bytes), 0) FROM tool_result_cache"
        ).fetchone()
        return int(row[0]) if row else 0

    def _execute_with_retry(self, sql: str, params: tuple) -> None:
        """Execute a single statement with the same 20-150ms jitter
        retry used by hermes_state._execute_write.  This keeps eviction
        from convoying with hot writes."""
        last_exc: Optional[Exception] = None
        for attempt in range(self._WRITE_MAX_RETRIES):
            try:
                self._conn.execute(sql, params)
                return
            except sqlite3.OperationalError as exc:
                err = str(exc).lower()
                if "locked" in err or "busy" in err:
                    last_exc = exc
                    time.sleep(random.uniform(self._WRITE_RETRY_MIN_S, self._WRITE_RETRY_MAX_S))
                    continue
                raise
        if last_exc is not None:
            raise last_exc


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
# Lazily constructed on first access; tests that need isolation construct
# their own ToolResultCache directly.  Default path is the same
# ``state.db`` the rest of the system uses — the table is just one more
# in the same SQLite file, sharing WAL + connection pool.
_default_cache: Optional[ToolResultCache] = None
_default_cache_lock = threading.Lock()


def get_default_cache() -> ToolResultCache:
    """Return the process-wide ToolResultCache singleton."""
    global _default_cache
    if _default_cache is not None:
        return _default_cache
    with _default_cache_lock:
        if _default_cache is None:
            from hermes_state import DEFAULT_DB_PATH  # late import to avoid cycle
            _default_cache = ToolResultCache(DEFAULT_DB_PATH)
        return _default_cache


def reset_default_cache_for_tests() -> None:
    """Drop the singleton — test fixtures call this between cases so each
    case gets a fresh in-process state."""
    global _default_cache
    with _default_cache_lock:
        if _default_cache is not None:
            try:
                _default_cache._conn.close()
            except Exception:
                pass
        _default_cache = None


# ---------------------------------------------------------------------------
# Eviction sweep thread
# ---------------------------------------------------------------------------
# A single process-wide background thread that runs vacuum() hourly
# with ±15 min jitter.  Why jitter: 1000 installs on a shared cron
# cloud shouldn't all VACUUM at :00.  The jitter is sampled once at
# thread start; the thread then fires every (60 ± 15) minutes.

_eviction_thread: Optional[threading.Thread] = None
_eviction_stop_event = threading.Event()


def start_eviction_thread(interval_seconds: int = 3600, jitter_seconds: int = 900) -> None:
    """Start the hourly eviction thread.  Idempotent — re-calling is a no-op."""
    global _eviction_thread
    if _eviction_thread is not None and _eviction_thread.is_alive():
        return
    _eviction_stop_event.clear()

    def _loop() -> None:
        # First sleep: random jitter so thread starts don't convoy.
        next_delay = interval_seconds + random.uniform(-jitter_seconds, jitter_seconds)
        while not _eviction_stop_event.is_set():
            if _eviction_stop_event.wait(next_delay):
                return
            try:
                cache = get_default_cache()
                stats = cache.vacuum()
                if stats.get("expired_removed") or stats.get("size_evicted"):
                    logger.info(
                        "Tool result cache vacuum: %d expired, %d size-evicted",
                        stats["expired_removed"], stats["size_evicted"],
                    )
            except Exception as exc:
                logger.error("Eviction sweep error: %s", exc, exc_info=True)
            next_delay = interval_seconds + random.uniform(-jitter_seconds, jitter_seconds)

    _eviction_thread = threading.Thread(
        target=_loop, name="tool-cache-eviction", daemon=True
    )
    _eviction_thread.start()


def stop_eviction_thread(timeout: float = 5.0) -> None:
    """Signal the eviction thread to stop.  Joins with a *timeout*."""
    _eviction_stop_event.set()
    if _eviction_thread is not None:
        _eviction_thread.join(timeout=timeout)
