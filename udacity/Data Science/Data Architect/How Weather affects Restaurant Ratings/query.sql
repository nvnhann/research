-- JSON
create or replace file format jsonformat type='JSON' strip_outer_array=true;
create or replace stage json_stage file_format = jsonformat;

create or replace table yelp_academic_dataset_business(recordjson variant);                                                                         
create or replace table yelp_academic_dataset_checkin(recordjson variant);
create or replace table yelp_academic_dataset_covid_features(recordjson variant);
create or replace table yelp_academic_dataset_review(recordjson variant);
create or replace table yelp_academic_dataset_tip(recordjson variant);
create or replace table yelp_academic_dataset_user(recordjson variant);

-- CSV

create or replace file format csvformat type='CSV' compression='auto' field_delimiter=',' record_delimiter = '\n'  skip_header=1 error_on_column_count_mismatch=true null_if = ('NULL', 'null') empty_field_as_null = true;
create or replace stage csv_stage file_format = csvformat;

create or replace table "precipitation"(
    "date" DATE,
    "precipitation" STRING,
    "precipitation_normal" STRING
);
create or replace table "temperature"(
    "date" DATE,
    "min" DOUBLE,
    "max" DOUBLE,
    "normal_min" DOUBLE,
    "normal_max" DOUBLE
);

-- STAGING to ODS

USE udacity.ODS;

create or replace table "precipitation"(
    "date" date primary key,
    "precipitation" string,
    "precipitation_normal" double
);

insert into "precipitation"(
    "date", "precipitation", "precipitation_normal"
)
select 
    "date",
    "precipitation",
    "precipitation_normal"
from 
    udacity.STAGING."precipitation";

create or replace table "temperature"(
    "date" date primary key,
    "min_temp" double,
    "max_temp" double,
    "normal_min" double,
    "normal_max" double
);

insert into "temperature"(
    "date", "min_temp", "max_temp", "normal_min", "normal_max"
)
select 
    "date" date,
    "min" double,
    "max" double,
    "normal_min" double,
    "normal_max" double
from 
    udacity.STAGING."temperature";

create or replace table "geography"(
    "geography_id" number identity primary key,
    "address" string,
    "latitude" double,
    "longitude" double,
    "postal_code" string,
    "city" string,
    "state" string
);

insert into "geography"("address", "latitude", "longitude", "postal_code", "city", "state")
select distinct
    RECORDJSON:address,
    RECORDJSON:latitude,
    RECORDJSON:longitude,
    RECORDJSON:postal_code,
    RECORDJSON:city,
    RECORDJSON:state
from 
    udacity.STAGING.YELP_ACADEMIC_DATASET_BUSINESS;

create or replace table "business"(
    "business_id" string primary key,
    "geography_id" number references udacity.ODS."geography"("geography_id"),
    "name" string,
    "stars" double
);

insert into "business"
select distinct
    RECORDJSON:business_id,
    g."geography_id",
    RECORDJSON:name,
    RECORDJSON:stars
from 
    udacity.STAGING.YELP_ACADEMIC_DATASET_BUSINESS as b
join 
    udacity.ODS."geography" as g
on
    RECORDJSON:city = g."city" and
    RECORDJSON:address = g."address" and
    RECORDJSON:latitude = g."latitude" and
    RECORDJSON:longitude = g."longitude" and
    RECORDJSON:state = g."state";

create or replace table "checkin"(
    "business_id" string primary key references udacity.ODS."business"("business_id"),
    "date" string
);

insert into "checkin"
select 
    RECORDJSON:business_id,
    RECORDJSON:date
from 
    udacity.STAGING.YELP_ACADEMIC_DATASET_CHECKIN;

create or replace table "customer"(
    "customer_id" string primary key,
    "average_stars" double,
    "fans" number,
    "review_count" number,
    "name" string
);

insert into "customer"
select 
    RECORDJSON:user_id,
    RECORDJSON:average_stars,
    RECORDJSON:fans,
    RECORDJSON:review_count,
    RECORDJSON:name
from 
    udacity.STAGING.YELP_ACADEMIC_DATASET_USER;

create or replace table "covid"(
  "business_id" string primary key references udacity.ODS."business"("business_id"),
  "covid_banner" string,
  "virtual_services" string,
  "delivery_or_takeout" string,
);

insert into "covid"
select 
    RECORDJSON:business_id,
    RECORDJSON:"Covid Banner",
    RECORDJSON:"Virtual Services Offered",
    RECORDJSON:"delivery or takeout",
from 
    udacity.STAGING.YELP_ACADEMIC_DATASET_COVID_FEATURES;

create or replace table "review"(
    "review_id" string primary key,
    "business_id" string references udacity.ODS."business"("business_id"),
    "date" date,
    "cool" number,
    "funny" number,
    "stars" double,
    "useful" double,
    "user_id" string references udacity.ODS."customer"("customer_id")
);

insert into "review"
select 
    RECORDJSON:review_id,
    RECORDJSON:business_id,
    RECORDJSON:date,
    RECORDJSON:cool,
    RECORDJSON:funny,
    RECORDJSON:stars,
    RECORDJSON:useful,
    RECORDJSON:user_id
from 
    udacity.STAGING.YELP_ACADEMIC_DATASET_REVIEW;

create or replace table "tip"(
  "business_id" string primary key references udacity.ODS."business"("business_id"),
  "compliment_count" number,
  "date" date,
  "user_id" string references udacity.ODS."customer"("customer_id")
);

insert into "tip"
select 
    RECORDJSON:business_id,
    RECORDJSON:compliment_count,
    RECORDJSON:date,
    RECORDJSON:user_id
from 
    udacity.STAGING.YELP_ACADEMIC_DATASET_TIP;

SELECT * 
	FROM UDACITY.ODS."precipitation" AS p
	JOIN UDACITY.ODS."review" AS r ON r."date" = p."date"
	JOIN UDACITY.ODS."temperature" AS t ON t."date" = r."date"
	JOIN UDACITY.ODS."business" AS b ON b."business_id" = r."business_id"
    JOIN UDACITY.ODS."covid" AS c ON b."business_id" = c."business_id"
	JOIN UDACITY.ODS."checkin" AS ch ON b."business_id" = ch."business_id"
	JOIN UDACITY.ODS."tip" AS x ON b."business_id" = x."business_id"
    JOIN UDACITY.ODS."customer" AS cus ON cus."customer_id" = r."user_id";

-- ODS to DW

create or replace table dim_customer(
    "customer_id" string primary key,
    "name" string,
    "average_stars" double,
    "fans" number,
    "review_count" number
);

insert into dim_customer
select distinct
    "customer_id",
    "name",
    "average_stars",
    "fans",
    "review_count"
from udacity.ODS."customer";
CREATE
	OR replace TABLE UDACITY.DW.dim_climate (
	"date" DATE PRIMARY KEY
	,"min_temp" DOUBLE
	,"max_temp" DOUBLE
	,"precipitation" string
	,"precipitation_normal" DOUBLE
	);

INSERT INTO UDACITY.DW.dim_climate
SELECT DISTINCT p."date"
	,t."min_temp"
	,t."max_temp"
	,p."precipitation"
	,p."precipitation_normal"
FROM udacity.ODS."precipitation" AS p
INNER JOIN udacity.ODS."temperature" AS t ON t."date" = p."date"

create or replace table dim_business(
    "business_id" string primary key,
    "name" string,
    "stars" double,
    "city" string,
    "state" string,
    "postal_code" string,
    "checkin_dates" string
);

insert into dim_business(
    "business_id",
    "name",
    "stars",
    "city",
    "state",
    "postal_code",
    "checkin_dates"
)
select
    b."business_id",
    b."name",
    b."stars",
    g."city",
    g."state",
    g."postal_code",
    ch."date"
from udacity.ODS."business" as b 
join udacity.ODS."geography" as g on b."geography_id" =  g."geography_id"
join udacity.ODS."checkin" as ch on b."business_id" = ch."business_id";

create or replace table fact_info(
    "fact_id" string primary key,
    "business_id" string references udacity.DW."DIM_BUSINESS"("business_id"),
    "customer_id" string references udacity.DW."DIM_CUSTOMER"("customer_id"),
    "date" date references udacity.DW."DIM_TEMPERATURE"("date" ),
    "stars" double
);

insert into fact_info
select
    r."review_id",
    r."business_id",
    r."user_id",
    r."date",
    r."stars"
from
    udacity.ODS."review" as r;

SELECT
    b."name" AS "Business Name",
    t."date" AS "Date",
    t."min_temp" AS "Min Temperature",
    t."max_temp" AS "Max Temperature",
    p."precipitation" AS "Precipitation",
    f."stars" AS "Ratings"
FROM
    fact_info AS f
JOIN
    dim_business AS b ON f."business_id" = b."business_id"
JOIN
    dim_temperature AS t ON f."date" = t."date"
JOIN
    dim_precipitation AS p ON f."date" = p."date";