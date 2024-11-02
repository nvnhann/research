USE externals;

CREATE EXTERNAL TABLE staging_trip (
	[trip_id] nvarchar(4000),
	[rideable_type] nvarchar(4000),
	[start_at] varchar(50),
	[ended_at] varchar(50),
	[start_station_id] nvarchar(4000),
	[end_station_id] nvarchar(4000),
	[rider_id] bigint
	)
	WITH (
	LOCATION = 'publictrip.csv',
	DATA_SOURCE = [AzureDataResource],
	FILE_FORMAT = [SynapseDelimitedTextFormat]
	)
GO

SELECT TOP 100 * FROM dbo.staging_trip
GO