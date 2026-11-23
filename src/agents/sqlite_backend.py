"""Read-Only SQLite/SQL Backend thực thi câu lệnh Text-to-SQL an toàn cục bộ."""

import sqlite3
from pathlib import Path
from typing import Any

from agents.sql_agent import SQLExecutionError


class ReadOnlySQLiteBackend:
    """Thực thi câu lệnh SELECT trên SQLite ở chế độ Read-Only (uri mode=ro)."""

    def __init__(self, db_path: Path) -> None:
        if not db_path.exists():
            raise FileNotFoundError(f"Không tìm thấy database: {db_path}")
        self._db_uri = f"file:{db_path.as_posix()}?mode=ro"

    async def execute_readonly(self, sql: str) -> list[dict[str, Any]]:
        """Chạy câu lệnh SELECT và trả về danh sách dict kết quả."""
        try:
            with sqlite3.connect(self._db_uri, uri=True, timeout=3.0) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(sql)
                return [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error as exc:
            raise SQLExecutionError(f"Lỗi SQL: {exc}") from exc


def seed_sample_financial_db(db_path: Path, schema_path: Path) -> None:
    """Khởi tạo DB mẫu với dữ liệu tài chính FPT, FOX, VCB, HPG để test Text-to-SQL."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    schema_sql = schema_path.read_text(encoding="utf-8")

    with sqlite3.connect(db_path) as conn:
        conn.executescript(schema_sql)
        existing = conn.execute("SELECT COUNT(*) FROM companies").fetchone()[0]
        if existing > 0:
            return

        conn.executemany(
            "INSERT INTO companies (ticker, company_name, exchange, sector) VALUES (?, ?, ?, ?)",
            [
                ("FPT", "Công ty Cổ phần FPT", "HOSE", "Công nghệ thông tin"),
                ("FOX", "Công ty Cổ phần Viễn thông FPT", "UPCOM", "Viễn thông"),
                ("VCB", "Ngân hàng TMCP Ngoại thương Việt Nam", "HOSE", "Ngân hàng"),
                ("HPG", "Công ty Cổ phần Tập đoàn Hòa Phát", "HOSE", "Thép"),
            ],
        )
        conn.executemany(
            """INSERT INTO quarterly_financials
            (ticker, year, quarter, revenue_billion_vnd, gross_profit_billion_vnd,
             net_profit_billion_vnd, roe_pct, eps_vnd)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                ("FPT", 2025, 1, 16058.0, 6210.0, 2174.0, 28.4, 1480.0),
                ("FPT", 2025, 2, 16920.0, 6540.0, 2310.0, 28.9, 1570.0),
                ("FOX", 2024, 4, 4850.0, 2120.0, 815.0, 29.1, 1650.0),
                ("VCB", 2025, 1, 15800.0, 15800.0, 8920.0, 21.5, 1590.0),
                ("HPG", 2025, 1, 34200.0, 4850.0, 3150.0, 11.8, 490.0),
            ],
        )
        conn.executemany(
            """INSERT INTO daily_stock_prices
            (ticker, trade_date, close_price_vnd, volume, pe_ratio, pb_ratio)
            VALUES (?, ?, ?, ?, ?, ?)""",
            [
                ("FPT", "2025-06-30", 132500.0, 4250000, 23.4, 5.8),
                ("VCB", "2025-06-30", 91200.0, 1850000, 14.8, 2.9),
                ("HPG", "2025-06-30", 28500.0, 24100000, 14.2, 1.6),
            ],
        )
        conn.commit()
