-- Schema cơ sở dữ liệu Tài chính - Chứng khoán Việt Nam (PostgreSQL / SQLite compatible)

CREATE TABLE IF NOT EXISTS companies (
    ticker VARCHAR(10) PRIMARY KEY,
    company_name TEXT NOT NULL,
    exchange VARCHAR(10) NOT NULL,
    sector TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS quarterly_financials (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker VARCHAR(10) NOT NULL REFERENCES companies(ticker),
    year INTEGER NOT NULL,
    quarter INTEGER NOT NULL,
    revenue_billion_vnd REAL NOT NULL,
    gross_profit_billion_vnd REAL NOT NULL,
    net_profit_billion_vnd REAL NOT NULL,
    roe_pct REAL NOT NULL,
    eps_vnd REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS daily_stock_prices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker VARCHAR(10) NOT NULL REFERENCES companies(ticker),
    trade_date DATE NOT NULL,
    close_price_vnd REAL NOT NULL,
    volume INTEGER NOT NULL,
    pe_ratio REAL NOT NULL,
    pb_ratio REAL NOT NULL
);
