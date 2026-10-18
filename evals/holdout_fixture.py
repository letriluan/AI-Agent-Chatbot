"""Independent synthetic evaluation fixture; no personal data."""
from contextlib import closing
import sqlite3


def create_database(path):
    with closing(sqlite3.connect(path)) as conn:
        conn.executescript("""
        CREATE TABLE customers (id INTEGER PRIMARY KEY, state TEXT);
        CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER REFERENCES customers(id), status TEXT, amount REAL, ordered_at TEXT);
        CREATE TABLE order_items (id INTEGER PRIMARY KEY, order_id INTEGER REFERENCES orders(id), quantity INTEGER, unit_price REAL);
        INSERT INTO customers VALUES (1,'SA'),(2,'WA'),(3,'TAS'),(4,'NT'),(5,'WA');
        INSERT INTO orders VALUES
        (1,1,'delivered',100,'2026-01-01'),(2,1,'cancelled',20,'2026-01-15'),
        (3,2,'delivered',80,'2026-02-01'),(4,2,'pending',0,'2026-02-28'),
        (5,3,'delivered',120,'2026-04-01'),(6,3,'delivered',120,'2026-04-02'),
        (7,5,'pending',60,'2026-04-30'),(8,5,'delivered',40,'2026-05-01');
        INSERT INTO order_items VALUES
        (1,1,2,10),(2,1,1,80),(3,2,1,20),(4,3,2,40),
        (5,5,4,30),(6,6,2,60),(7,7,2,30),(8,8,2,20);
        """)
        conn.commit()
