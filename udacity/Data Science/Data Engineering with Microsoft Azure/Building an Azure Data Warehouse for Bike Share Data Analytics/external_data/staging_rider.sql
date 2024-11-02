USE externals;

CREATE EXTERNAL TABLE dbo.staging_rider (
	[Rider_Id] bigint,
	[Address] nvarchar(4000),
	[First_Name] nvarchar(4000),
	[Last_Name] nvarchar(4000),
	[Birthday] varchar(50),
	[Account_start_date] varchar(50),
	[Account_end_date] varchar(50),
	[Is_member] bit
	)
	WITH (
	LOCATION = 'publicrider.csv',
	DATA_SOURCE = [AzureDataResource],
	FILE_FORMAT = [SynapseDelimitedTextFormat]
	)
GO

SELECT TOP 100 * FROM dbo.staging_rider
GO