USE externals;

CREATE EXTERNAL TABLE staging_station (
	[station_id] nvarchar(4000),
	[name] nvarchar(4000),
	[latitude] float,
	[longitude] float
	)
	WITH (
	LOCATION = 'publicstation.csv',
	DATA_SOURCE = [AzureDataResource],
	FILE_FORMAT = [SynapseDelimitedTextFormat]
	)
GO

SELECT TOP 100 * FROM dbo.staging_station
GO