"""Small synthetic public fixture: no real customer data."""
from contextlib import closing
import sqlite3


def create_database(path):
    with closing(sqlite3.connect(path)) as conn:
        conn.executescript("""
        CREATE TABLE customers (id INTEGER PRIMARY KEY, state TEXT);
        CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER REFERENCES customers(id), status TEXT, amount REAL, ordered_at TEXT);
        INSERT INTO customers VALUES (1,'SA'),(2,'WA'),(3,'TAS'),(4,'NT');
        INSERT INTO orders VALUES
        (1,1,'delivered',10,'2026-01-01'),
        (2,1,'delivered',20,'2026-01-02'),
        (3,2,'pending',30,'2026-02-01'),
        (4,2,'delivered',40,'2026-02-02'),
        (5,3,'cancelled',50,'2026-03-01'),
        (6,3,'delivered',60,'2026-03-02');
        """)
        conn.commit()
