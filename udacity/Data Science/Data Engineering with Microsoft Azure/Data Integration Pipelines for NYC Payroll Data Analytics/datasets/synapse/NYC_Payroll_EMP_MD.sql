IF NOT EXISTS (SELECT * FROM sys.external_file_formats WHERE name = 'SynapseDelimitedTextFormat') 
	CREATE EXTERNAL FILE FORMAT [SynapseDelimitedTextFormat] 
	WITH ( FORMAT_TYPE = DELIMITEDTEXT ,
	       FORMAT_OPTIONS (
			 FIELD_TERMINATOR = ',',
			 FIRST_ROW = 1,
			 USE_TYPE_DEFAULT = FALSE
			))
GO

IF NOT EXISTS (SELECT * FROM sys.external_data_sources WHERE name = 'dirpayrollfiles_adlsnycpayrolastintial_dfs_core_windows_net') 
	CREATE EXTERNAL DATA SOURCE [dirpayrollfiles_adlsnycpayrolastintial_dfs_core_windows_net] 
	WITH (
		LOCATION = 'abfss://dirpayrollfiles@adlsnycpayrolastintial.dfs.core.windows.net' 
	)
GO

CREATE EXTERNAL TABLE [dbo].[NYC_Payroll_EMP_MD] (
	[EmployeeID] nvarchar(4000),
	[LastName] nvarchar(4000),
	[FirstName] nvarchar(4000)
	)
	WITH (
	LOCATION = 'EmpMaster.csv',
	DATA_SOURCE = [dirpayrollfiles_adlsnycpayrolastintial_dfs_core_windows_net],
	FILE_FORMAT = [SynapseDelimitedTextFormat]
	)
GO


SELECT TOP 100 * FROM [dbo].[NYC_Payroll_EMP_MD]
GO