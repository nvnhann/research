# Project submission for Udacity Data Engineering Nanodegree - Data Warehouse

## Project summary

In order to facilitate analytics, this project merges music listen log files with song metadata. The Python is used to construct a Redshift cluster, and SQL and Python data pipelines are used to prepare a data schema for analytics. Before being added to a star schema with fact and dimension tables, JSON data is copied from an S3 bucket to Redshift staging tables. The songplays fact database offers simple analytics queries, while the four dimension tables—users, songs, artists, and time—make it simple to retrieve additional fields. Because aggregations are quick, queries can be kept straightforward, and denormalization is simple, a star schema makes sense for this application.

## Project Instructions

---

# Schema for Song Play Analysis

## Staging Tables

### `staging_events`
This table is used as an intermediate storage area for data ingested from user activity logs. It includes details about the actions users take while interacting with the music streaming service.

- **Columns**: artist, auth, firstName, gender, itemInSession, lastName, length, level, location, method, page, registration, sessionId, song, status, ts, userAgent, userId.

### `staging_songs`
This table serves as a temporary storage for metadata about songs and artists, ingested from the music dataset.

- **Columns**: num_songs, artist_id, artist_latitude, artist_longitude, artist_location, artist_name, song_id, title, duration, year.

## Fact Table

### `songplays`
This table records the fact of song plays in the app and is used to analyze the app's usage. Each row represents a song played by a user at a given time, linking to dimensions that describe who, what, and where.

- **Columns**: songplay_id, start_time, user_id, level, song_id, artist_id, session_id, location, user_agent.
- **Distribution Style**: Keyed on start_time for query optimization.
- **Primary Key**: songplay_id.

## Dimension Tables

### `users`
Stores information about users of the music streaming app.

- **Columns**: user_id, first_name, last_name, gender, level.
- **Distribution Style**: All, to improve performance on user-related queries.
- **Primary Key**: user_id.

### `songs`
Contains details about songs available in the app.

- **Columns**: song_id, title, artist_id, year, duration.
- **Distribution Style**: All, ensuring fast access to song information.
- **Primary Key**: song_id.

### `artists`
Holds information about the artists of the songs.

- **Columns**: artist_id, name, location, latitude, longitude.
- **Distribution Style**: All, facilitating efficient artist-related queries.
- **Primary Key**: artist_id.

### `time`
A table to break down timestamps of records into specific units of time, aiding in time-based analysis.

- **Columns**: start_time, hour, day, week, month, year, weekday.
- **Distribution Style**: Keyed on start_time to enhance performance on time-related queries.
- **Primary Key**: start_time.

---
### File

```shell
.
├── create_table.py
├── dwh.cfg
├── etl.py
├── requirements.txt
└── sql_queries.py
```

### Edit config in ```dwh.cfg```

```config
[CLUSTER]
HOST=
DB_NAME=
DB_USER=
DB_PASSWORD=
DB_PORT=

[IAM_ROLE]
ARN=

[S3]
LOG_DATA=
LOG_JSONPATH=
SONG_DATA=

[AWS]
REGION=
```

### install 
```shell
$ pip install -r requirements.txt
```

The files required for submission include ```sql_queries.py```, ```create_tables.py``` and ```etl.py```


Following the generation of ```dwh.cfg```, table creation and the execution of the ETL process are carried out using the provided Python scripts.
Drop and recreate tables

```python
python create_tables.py
```
Run ETL pipeline

```python
python etl.py
```