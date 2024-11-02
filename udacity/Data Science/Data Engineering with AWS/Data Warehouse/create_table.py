"""
This module defines functions to drop and create tables in a PostgreSQL database.
It reads database configuration from 'dwh.cfg', connects to the database,
drops tables if they exist, and then creates tables according to the specified SQL queries.
"""

import configparser
import psycopg2
from sql_queries import create_table_queries, drop_table_queries


def drop_tables(cur, conn):
    """
    Drops each table using the queries in `drop_table_queries` list.
    Parameters:
    - cur: Cursor of the database connection to execute PostgreSQL commands.
    - conn: Database connection to commit the changes.
    """
    for query in drop_table_queries:
        cur.execute(query)
        conn.commit()


def create_tables(cur, conn):
    """
    Creates tables using the queries in `create_table_queries` list.
    Each query is printed before execution. After all tables are created,
    a success message is printed.
    Parameters:
    - cur: Cursor of the database connection to execute PostgreSQL commands.
    - conn: Database connection to commit the changes.
    """
    for query in create_table_queries:
        print("Query: {}".format(query))
        cur.execute(query)
        conn.commit()
    print ("Create table successfully!")


def main():
    """
    Main function to manage the workflow of connecting to the PostgreSQL database,
    dropping existing tables, creating new tables, and closing the connection.
    The database configurations are read from 'dwh.cfg'.
    """
    config = configparser.ConfigParser()
    config.read('dwh.cfg')
    conn = psycopg2.connect("host={} dbname={} user={} password={} port={}".format(*config['CLUSTER'].values()))
    print("Connection successfully!")
    cur = conn.cursor()

    drop_tables(cur, conn)
    create_tables(cur, conn)
    conn.close()


if __name__ == "__main__":
    main()