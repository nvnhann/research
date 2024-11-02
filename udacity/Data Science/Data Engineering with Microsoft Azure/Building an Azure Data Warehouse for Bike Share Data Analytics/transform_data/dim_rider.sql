IF OBJECT_ID('dbo.dim_rider') IS NOT NULL
BEGIN
    DROP EXTERNAL TABLE [dbo].[dim_rider]
END
CREATE EXTERNAL TABLE dbo.dim_rider 
WITH
( 
	LOCATION = 'dimrider',
    DATA_SOURCE = [AzureDataResource],
    FILE_FORMAT = [SynapseDelimitedTextFormat]
)
AS
SELECT 
    [Rider_Id]
    ,[First_Name]
    ,[Last_Name]
    ,[Address]
    ,[Birthday]
    ,[Account_start_date]
    ,[Account_end_date]
    ,[Is_member]
FROM dbo.staging_rider;

SELECT TOP 10 * FROM dbo.dim_rider;