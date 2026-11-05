"""Invented transit star schema, not an Adelaide Metro feed or real service data.

Grain: one scheduled trip/stop event on a service date. Time is seconds from
service-day midnight, so 25:05 is valid and is not wrapped into a different day.
"""
from contextlib import closing
import sqlite3


def create_database(path):
    with closing(sqlite3.connect(path)) as conn:
        conn.executescript("""
        PRAGMA foreign_keys=ON;
        CREATE TABLE dim_route (route_key TEXT PRIMARY KEY, route_short_name TEXT, route_type INTEGER);
        CREATE TABLE dim_stop (stop_key TEXT PRIMARY KEY, stop_name TEXT, zone TEXT);
        CREATE TABLE dim_date (date_key INTEGER PRIMARY KEY, service_date TEXT, weekday TEXT);
        CREATE TABLE dim_time (time_key INTEGER PRIMARY KEY, service_hour INTEGER, peak_period TEXT);
        CREATE TABLE fact_stop_events (
            event_key INTEGER PRIMARY KEY, trip_id TEXT,
            route_key TEXT REFERENCES dim_route(route_key),
            stop_key TEXT REFERENCES dim_stop(stop_key),
            date_key INTEGER REFERENCES dim_date(date_key),
            time_key INTEGER REFERENCES dim_time(time_key),
            stop_sequence INTEGER, arrival_seconds INTEGER, departure_seconds INTEGER);
        INSERT INTO dim_route VALUES ('R1','B10',3),('R2','T20',0),('R3','R30',2),('R4','B40',3);
        INSERT INTO dim_stop VALUES
            ('S1','Central','CBD'),('S2','North','Suburban'),('S3','Campus','Suburban'),
            ('S4','Beach','Coastal'),('S5','Airport','Outer'),('S6','Unused','Outer');
        INSERT INTO dim_date VALUES
            (20261102,'2026-11-02','Monday'),(20261103,'2026-11-03','Tuesday'),
            (20261107,'2026-11-07','Saturday'),(20261108,'2026-11-08','Sunday');
        INSERT INTO dim_time VALUES (8,8,'peak'),(9,9,'off_peak'),(10,10,'off_peak'),(25,25,'off_peak');
        INSERT INTO fact_stop_events VALUES
            (1,'T1','R1','S1',20261102,8,1,28800,28860),
            (2,'T1','R1','S2',20261102,8,2,29400,29400),
            (3,'T1','R1','S3',20261102,8,3,30300,30300),
            (4,'T2','R1','S1',20261102,9,1,32400,32520),
            (5,'T2','R1','S2',20261102,9,2,33120,33120),
            (6,'T2','R1','S3',20261102,9,3,34200,34200),
            (7,'T3','R1','S1',20261103,8,1,28800,28860),
            (8,'T3','R1','S2',20261103,8,2,29400,29400),
            (9,'T4','R2','S1',20261102,8,1,28800,28800),
            (10,'T4','R2','S4',20261102,8,2,30000,30000),
            (11,'T5','R2','S1',20261107,10,1,36000,36000),
            (12,'T5','R2','S4',20261107,10,2,37200,37200),
            (13,'T6','R3','S1',20261102,25,1,90300,90300),
            (14,'T6','R3','S5',20261102,25,2,91500,91500);
        """)
        conn.commit()
