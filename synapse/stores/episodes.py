"""Episodic memory: the episodes table and the searches over it."""

from __future__ import annotations

import json
import re
import sqlite3
import time
import uuid
from typing import Any, Dict, List, Optional, Union


class EpisodeStore:
    """Episodes, their full-text search, and consolidation candidates."""

    def add_episode(
        self,
        title: str,
        situation: str = "",
        action_taken: str = "",
        outcome: str = "",
        lessons_learned: str = "",
        salience: float = 0.5,
        confidence: float = 1.0,
        tags: Optional[Union[str, List[str]]] = None,
        episode_id: Optional[str] = None,
    ) -> str:
        """
        Add a new structured episodic memory (Situation -> Action -> Outcome -> Lesson).
        Returns the generated or provided episode_id.
        """
        uid = episode_id or uuid.uuid4().hex
        now = int(time.time())

        tags_str = ""
        if tags is not None:
            if isinstance(tags, list):
                tags_str = json.dumps(tags)
            else:
                tags_str = str(tags).strip()

        with self._write():
            self._conn.execute(
                """
                INSERT INTO episodes (
                    id, title, situation, action_taken, outcome, lessons_learned,
                    salience, confidence, created_at, last_accessed, tags
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    uid,
                    title,
                    situation or "",
                    action_taken or "",
                    outcome or "",
                    lessons_learned or "",
                    float(salience),
                    float(confidence),
                    now,
                    now,
                    tags_str,
                ),
            )
            self._conn.commit()
        return uid

    def get_episode(self, episode_id: str, touch: bool = False) -> Optional[Dict[str, Any]]:
        """
        Retrieve an episodic memory by ID.
        Optionally updates last_accessed timestamp when touch=True.
        """
        with self._write():
            cursor = self._conn.execute(
                "SELECT * FROM episodes WHERE id = ?",
                (episode_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return None

            ep = dict(row)
            if touch:
                now = int(time.time())
                self._conn.execute(
                    "UPDATE episodes SET last_accessed = ? WHERE id = ?",
                    (now, ep["id"]),
                )
                self._conn.commit()
                ep["last_accessed"] = now

            return ep

    def update_episode(self, episode_id: str, **kwargs: Any) -> bool:
        """
        Update fields of an episodic memory by ID.
        Returns True if a row was updated, False otherwise.
        """
        if not kwargs:
            return False

        allowed_fields = {
            "title",
            "situation",
            "action_taken",
            "outcome",
            "lessons_learned",
            "salience",
            "confidence",
            "tags",
            "created_at",
            "last_accessed",
        }

        updates: Dict[str, Any] = {}
        for k, v in kwargs.items():
            if k in allowed_fields:
                if k == "tags" and isinstance(v, list):
                    updates[k] = json.dumps(v)
                elif k in ("salience", "confidence"):
                    updates[k] = float(v)
                else:
                    updates[k] = v

        if not updates:
            return False

        if "last_accessed" not in updates:
            updates["last_accessed"] = int(time.time())

        set_clause = ", ".join(f"{col} = ?" for col in updates.keys())
        values = list(updates.values())
        values.append(episode_id)

        with self._write():
            cursor = self._conn.execute(
                f"UPDATE episodes SET {set_clause} WHERE id = ?",
                values,
            )
            self._conn.commit()
            return cursor.rowcount > 0

    def delete_episode(self, episode_id: str) -> bool:
        """
        Delete an episodic memory by ID.
        Triggers cascade removal from episodes_fts.
        Returns True if deleted, False otherwise.
        """
        with self._write():
            cursor = self._conn.execute(
                "DELETE FROM episodes WHERE id = ?",
                (episode_id,),
            )
            self._conn.commit()
            return cursor.rowcount > 0

    def list_episodes(
        self,
        limit: int = 100,
        min_salience: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        List episodes ordered by salience and recency.
        """
        with self._write():
            cursor = self._conn.execute(
                """
                SELECT * FROM episodes
                WHERE salience >= ?
                ORDER BY salience DESC, last_accessed DESC, created_at DESC
                LIMIT ?
                """,
                (float(min_salience), max(1, int(limit))),
            )
            return [dict(row) for row in cursor.fetchall()]

    def search_episodes(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Execute full-text search against episodes_fts indexed fields
        (title, situation, action_taken, outcome, lessons_learned, tags)
        ranked by BM25 relevance score.
        """
        if not query or not query.strip():
            return []

        tokens = re.findall(r"[\w\-]+", query)
        if not tokens:
            return []

        formatted_query = self._sanitize_fts_query(query)
        if not formatted_query:
            return []

        with self._write():
            try:
                cursor = self._conn.execute(
                    """
                    SELECT e.*, f.rank
                    FROM episodes_fts f
                    JOIN episodes e ON e.id = f.id
                    WHERE episodes_fts MATCH ?
                    ORDER BY f.rank ASC, e.salience DESC
                    LIMIT ?
                    """,
                    (formatted_query, max(1, int(limit))),
                )
                results = [dict(row) for row in cursor.fetchall()]
                if not results and len(tokens) > 1:
                    or_query = " OR ".join(f'"{t}"*' for t in tokens)
                    cursor = self._conn.execute(
                        """
                        SELECT e.*, f.rank
                        FROM episodes_fts f
                        JOIN episodes e ON e.id = f.id
                        WHERE episodes_fts MATCH ?
                        ORDER BY f.rank ASC, e.salience DESC
                        LIMIT ?
                        """,
                        (or_query, max(1, int(limit))),
                    )
                    results = [dict(row) for row in cursor.fetchall()]
                return results
            except sqlite3.OperationalError:
                cursor = self._conn.execute(
                    """
                    SELECT e.*, f.rank
                    FROM episodes_fts f
                    JOIN episodes e ON e.id = f.id
                    WHERE episodes_fts MATCH ?
                    ORDER BY f.rank ASC, e.salience DESC
                    LIMIT ?
                    """,
                    (f'"{query}"', max(1, int(limit))),
                )
                return [dict(row) for row in cursor.fetchall()]

    def list_consolidatable_episodes(self, min_salience: float = 0.5) -> List[Dict[str, Any]]:
        """Episodes that carry usable lessons and meet the salience floor."""
        with self._write():
            cursor = self._conn.execute(
                """
                SELECT * FROM episodes
                WHERE salience >= ?
                  AND lessons_learned IS NOT NULL
                  AND trim(lessons_learned) != ''
                ORDER BY salience DESC, last_accessed DESC, created_at DESC
                """,
                (float(min_salience),),
            )
            return [dict(row) for row in cursor.fetchall()]
