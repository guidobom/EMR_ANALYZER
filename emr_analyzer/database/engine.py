"""SQLite database engine for EMR Analyzer.

Manages connections, enforces WAL mode, and provides context manager
for transactions.
"""

import sqlite3
import threading
from pathlib import Path


class DatabaseEngine:
    """SQLite connection manager with WAL mode and thread safety."""

    def __init__(self, db_path: Path):
        self._db_path = db_path
        self._local = threading.local()

    def _ensure_dir(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._ensure_dir()
            conn = sqlite3.connect(str(self._db_path))
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            # A desktop installation can have GUI and worker threads touching
            # the same database. Wait briefly for the writer instead of
            # surfacing a transient "database is locked" error.
            conn.execute("PRAGMA busy_timeout=10000")
            # WAL + NORMAL is the recommended durability/performance balance
            # for a local application: committed data remains durable while
            # avoiding a full disk sync for every small write.
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return self._local.conn

    def close(self) -> None:
        """Close the thread-local connection if open."""
        if hasattr(self._local, "conn") and self._local.conn is not None:
            self._local.conn.close()
            self._local.conn = None
        self._local.transaction_depth = 0

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        """Execute a SQL statement and return the cursor."""
        return self.connection.execute(sql, params)

    def executemany(self, sql: str, params_list: list[tuple]) -> sqlite3.Cursor:
        """Execute a SQL statement with multiple parameter sets."""
        return self.connection.executemany(sql, params_list)

    def commit(self) -> None:
        """Commit unless an enclosing managed transaction owns the boundary.

        Repository methods may commit when called alone, but must not release
        an outer transaction (or its savepoints) when composed by a service.
        """
        if getattr(self._local, "transaction_depth", 0):
            return
        self.connection.commit()

    def rollback(self) -> None:
        """Rollback the current transaction."""
        self.connection.rollback()

    def __enter__(self):
        depth = getattr(self._local, "transaction_depth", 0)
        if depth == 0:
            if not self.connection.in_transaction:
                self.connection.execute("BEGIN")
        else:
            self.connection.execute(f"SAVEPOINT emr_nested_{depth}")
        self._local.transaction_depth = depth + 1
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        depth = getattr(self._local, "transaction_depth", 1) - 1
        self._local.transaction_depth = max(0, depth)
        if depth == 0:
            if exc_type is None:
                self.commit()
            else:
                self.rollback()
        elif exc_type is None:
            self.connection.execute(f"RELEASE SAVEPOINT emr_nested_{depth}")
        else:
            self.connection.execute(f"ROLLBACK TO SAVEPOINT emr_nested_{depth}")
            self.connection.execute(f"RELEASE SAVEPOINT emr_nested_{depth}")
        return False

    def vacuum(self) -> None:
        """Optimize database file size."""
        self.execute("VACUUM")

    def get_table_count(self, table: str, where: str = "",
                        params: tuple = ()) -> int:
        """Count rows in a table with optional WHERE clause."""
        sql = f"SELECT COUNT(*) FROM {table}"
        if where:
            sql += f" WHERE {where}"
        cursor = self.execute(sql, params)
        return cursor.fetchone()[0]
