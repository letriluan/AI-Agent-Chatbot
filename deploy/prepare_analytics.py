"""Build-time indexes for the Olist demo, separate from generic CSV ingestion."""
import argparse
import sqlite3


def prepare_analytics(conn):
    # Stream months in grouping order and read item prices without table lookups.
    conn.execute("""CREATE INDEX IF NOT EXISTS idx_demo_orders_month_id
        ON olist_orders_dataset(strftime('%Y-%m', order_purchase_timestamp), order_id)""")
    conn.execute("""CREATE INDEX IF NOT EXISTS idx_demo_items_order_price
        ON olist_order_items_dataset(order_id, price)""")
    conn.execute('ANALYZE')
    conn.commit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', required=True)
    args = parser.parse_args()
    with sqlite3.connect(args.db) as connection:
        prepare_analytics(connection)
    print('Prepared Olist monthly-aggregation indexes.')
