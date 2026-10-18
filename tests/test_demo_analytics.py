"""Monthly aggregation retains all statuses and item prices, excluding freight."""
import sqlite3

from deploy.prepare_analytics import prepare_analytics


def test_monthly_indexes_preserve_join_totals_and_avoid_temporary_grouping():
    with sqlite3.connect(':memory:') as conn:
        conn.executescript('''
            CREATE TABLE olist_orders_dataset (
                order_id TEXT PRIMARY KEY, order_purchase_timestamp TEXT, order_status TEXT);
            CREATE TABLE olist_order_items_dataset (
                order_id TEXT, price REAL, freight_value REAL);
        ''')
        orders = [(f'o{i}', f'2020-{i % 12 + 1:02d}-01', 'delivered' if i % 2 else 'canceled') for i in range(240)]
        conn.executemany('INSERT INTO olist_orders_dataset VALUES (?,?,?)', orders)
        conn.executemany('INSERT INTO olist_order_items_dataset VALUES (?,?,?)',
                         [(f'o{i}', price, 999) for i in range(240) for price in (10, 20)])
        query = """SELECT strftime('%Y-%m', o.order_purchase_timestamp) AS month,
                   SUM(oi.price) AS total_price FROM olist_orders_dataset o
                   JOIN olist_order_items_dataset oi ON oi.order_id=o.order_id
                   GROUP BY month ORDER BY month ASC LIMIT 1000"""
        expected = conn.execute(query).fetchall()
        prepare_analytics(conn)
        prepare_analytics(conn)
        assert conn.execute(query).fetchall() == expected
        assert len(expected) == 12 and sum(row[1] for row in expected) == 7200
        plan = ' '.join(row[3] for row in conn.execute('EXPLAIN QUERY PLAN ' + query))
        assert 'idx_demo_orders_month_id' in plan
        assert 'COVERING INDEX idx_demo_items_order_price' in plan
        assert 'TEMP B-TREE' not in plan
