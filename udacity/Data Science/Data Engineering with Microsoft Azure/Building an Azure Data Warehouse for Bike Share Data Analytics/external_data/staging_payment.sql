USE externals;

CREATE EXTERNAL TABLE staging_payment (
	[payment_id] bigint,
	[date] varchar(50),
	[amount] float,
	[rider_id] bigint
	)
	WITH (
	LOCATION = 'publicpayment.csv',
	DATA_SOURCE = [AzureDataResource],
	FILE_FORMAT = [SynapseDelimitedTextFormat]
	)
GO

SELECT TOP 100 * FROM dbo.staging_payment
GO