"""Build-time indexes preserve answers and support filtered one-to-many joins."""
import sqlite3
import pandas as pd
from load_db import create_tables, insert_all, create_query_indexes


def test_covering_filter_and_foreign_key_indexes_preserve_distinct_count():
    tables = {
        'orders': pd.DataFrame({'order_id': [f'o{i}' for i in range(200)], 'status': ['delivered' if i % 2 else 'pending' for i in range(200)]}),
        'items': pd.DataFrame({'order_id': [f'o{i}' for i in range(200) for _ in range(2)], 'quantity': [1]*400}),
    }
    pks = {'orders': 'order_id', 'items': None}
    fks = [('items', 'order_id', 'orders', 'order_id')]
    with sqlite3.connect(':memory:') as conn:
        create_tables(conn, tables, pks, fks)
        insert_all(conn, tables)
        sql = "SELECT COUNT(DISTINCT o.order_id) FROM orders o JOIN items i ON i.order_id=o.order_id WHERE o.status='delivered'"
        expected = conn.execute(sql).fetchone()
        create_query_indexes(conn, tables, pks, fks)
        create_query_indexes(conn, tables, pks, fks)
        assert conn.execute(sql).fetchone() == expected == (100,)
        assert conn.execute('PRAGMA foreign_key_check').fetchall() == []
        indexes = conn.execute("SELECT name FROM sqlite_master WHERE type='index'").fetchall()
        assert ('idx_items_order_id',) in indexes
        assert ('idx_orders_status_order_id',) in indexes
        plan = ' '.join(row[3] for row in conn.execute('EXPLAIN QUERY PLAN '+sql))
        assert 'COVERING INDEX idx_orders_status_order_id' in plan
        assert 'COVERING INDEX idx_items_order_id' in plan
