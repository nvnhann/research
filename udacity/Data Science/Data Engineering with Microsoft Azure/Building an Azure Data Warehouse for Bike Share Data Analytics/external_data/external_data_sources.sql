USE externals;

IF NOT EXISTS (SELECT * FROM sys.external_data_sources WHERE name = 'AzureDataResource') 
	CREATE EXTERNAL DATA SOURCE [AzureDataResource] 
	WITH (
		LOCATION = 'abfss://filestorage@nhannv13.blob.core.windows.net'
	);
GO