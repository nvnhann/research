IF NOT EXISTS (SELECT * FROM sys.external_file_formats WHERE name = 'SynapseDelimitedTextFormat') 
	CREATE EXTERNAL FILE FORMAT [SynapseDelimitedTextFormat] 
	WITH ( FORMAT_TYPE = DELIMITEDTEXT ,
	       FORMAT_OPTIONS (
			 FIELD_TERMINATOR = ',',
			 USE_TYPE_DEFAULT = FALSE
			))
GO

IF NOT EXISTS (SELECT * FROM sys.external_data_sources WHERE name = 'nyc_adlsnycpayrolastintial_dfs_core_windows_net') 
	CREATE EXTERNAL DATA SOURCE [nyc_adlsnycpayrolastintial_dfs_core_windows_net] 
	WITH (
		LOCATION = 'abfss://nyc@adlsnycpayrolastintial.dfs.core.windows.net' 
	)
GO

CREATE EXTERNAL TABLE [dbo].[NYC_Payroll_Summary] (
	[FiscalYear] nvarchar(4000),
	[AgencyName] nvarchar(4000),
	[TotalPaid] nvarchar(4000)
	)
	WITH (
	LOCATION = 'part-00000-f8ef7e62-5c45-454b-ae7c-704b8c6974c7-c000.csv',
	DATA_SOURCE = [nyc_adlsnycpayrolastintial_dfs_core_windows_net],
	FILE_FORMAT = [SynapseDelimitedTextFormat]
	)
GO


SELECT TOP 100 * FROM [dbo].[NYC_Payroll_Summary]
GO