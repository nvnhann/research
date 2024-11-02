-- CREATE dim_payment TABLE
IF OBJECT_ID('dbo.fact_trip') IS NOT NULL
BEGIN
    DROP EXTERNAL TABLE dbo.fact_trip
END

CREATE EXTERNAL TABLE [dbo].[fact_trip] WITH(
    LOCATION = 'fact_trip',
    DATA_SOURCE = [AzureDataResource],
    FILE_FORMAT = [SynapseDelimitedTextFormat]
) AS (
    SELECT
        st.trip_id,
        st.rideable_type,
        st.rider_id,
        st.start_at,
        st.ended_at,
        st.start_station_id,
        st.end_station_id,
        DATEDIFF(HOUR, st.start_at, st.ended_at) AS Duration,
        DateDIFF(YEAR, sr.Birthday, st.start_at) AS Rider_Age
    FROM
        dbo.staging_trip AS st
        JOIN dbo.staging_rider AS sr
        ON sr.Rider_Id = st.rider_id
);

Go
SELECT
    TOP 10*
FROM 
    [dbo].[fact_trip];