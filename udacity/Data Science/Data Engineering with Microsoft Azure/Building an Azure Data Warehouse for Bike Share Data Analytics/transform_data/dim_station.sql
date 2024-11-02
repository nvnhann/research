--Create dim_station table
IF OBJECT_ID('dbo.dim_station') IS NOT NULL
BEGIN
    DROP EXTERNAL TABLE dbo.dim_station;
END

CREATE EXTERNAL TABLE dbo.dim_station 
WITH
( 
	LOCATION = 'dim_station',
    DATA_SOURCE = [AzureDataResource],
    FILE_FORMAT = [SynapseDelimitedTextFormat]
)
AS
SELECT 
	[station_id],
	[name],
	[latitude],
	[longitude]
FROM 
	staging_station;

SELECT TOP 10 * FROM dbo.dim_station;