"""
This module defines functions to load data into staging tables and insert data into analytics tables.
It reads SQL queries from 'sql_queries.py', connects to the database,
executes the queries, and commits the changes.
"""

import configparser
import psycopg2
from sql_queries import copy_table_queries, insert_table_queries


def load_staging_tables(cur, conn):
    """
    Loads data into staging tables.
    """
    for query in copy_table_queries:
        print("Query loading: {}".format(query))
        cur.execute(query)
        conn.commit()


def insert_tables(cur, conn):
    """
    Inserts data into analytics tables.
    """
    for query in insert_table_queries:
        print("Query insert: {}".format(query))
        cur.execute(query)
        conn.commit()


def main():
    """
    Main.
    """
    config = configparser.ConfigParser()
    config.read('dwh.cfg')

    conn = psycopg2.connect("host={} dbname={} user={} password={} port={}".format(*config['CLUSTER'].values()))
    cur = conn.cursor()
    
    load_staging_tables(cur, conn)
    insert_tables(cur, conn)
    print("Connection successfully")
    conn.close()


if __name__ == "__main__":
    main()