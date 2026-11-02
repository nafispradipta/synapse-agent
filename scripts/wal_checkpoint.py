#!/usr/bin/env python3
"""
WAL Checkpoint Scheduler for Synapse Memory Engine.
Runs PRAGMA wal_checkpoint(TRUNCATE) to fold the write-ahead log back into
the main database file, keeping the -wal file small and reads fast.
"""

import logging
import os
import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = Path(os.environ.get("SYNAPSE_DB_PATH") or (PROJECT_ROOT / "data" / "synapse.db")).resolve()
LOGS_DIR = Path(os.environ.get("SYNAPSE_LOGS_DIR") or (PROJECT_ROOT / "logs")).resolve()
LOGS_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOGS_DIR / "wal_checkpoint.log"

logging.basicConfig(filename=str(LOG_PATH), level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("wal_checkpoint")


def main() -> None:
    if not DB_PATH.exists():
        logger.error("Synapse DB not found at %s", DB_PATH)
        raise SystemExit(1)
    conn = sqlite3.connect(str(DB_PATH))
    try:
        row = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        busy, log, checkpointed = row
        logger.info("wal_checkpoint(TRUNCATE): busy=%d log=%d checkpointed=%d",
                    busy, log, checkpointed)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
