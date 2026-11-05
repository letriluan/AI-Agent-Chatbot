"""Fresh synthetic identifier/name conflict fixture; original transit data is unchanged."""
from contextlib import closing
import sqlite3

from .transit_fixture import create_database as create_transit


def create_database(path):
    create_transit(path)
    with closing(sqlite3.connect(path)) as conn:
        # S4 is both a real key and another stop's display name. Explicit user
        # intent must decide the column; a name may also contain a number.
        conn.execute("UPDATE dim_stop SET stop_name='S4' WHERE stop_key='S2'")
        conn.execute("INSERT INTO dim_stop VALUES ('S7','Platform 12','Outer')")
        conn.commit()
