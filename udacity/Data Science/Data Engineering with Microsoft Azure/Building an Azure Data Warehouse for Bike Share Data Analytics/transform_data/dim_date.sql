-- CREATE dim_payment TABLE
IF OBJECT_ID('dbo.dim_date') IS NOT NULL
BEGIN
    DROP EXTERNAL TABLE dbo.dim_date
END

CREATE EXTERNAL TABLE [dbo].[dim_payment] WITH(
    LOCATION = 'dim_date',
	DATA_SOURCE = [AzureDataResource],
	FILE_FORMAT = [SynapseDelimitedTextFormat]
) AS (
    SELECT
        ROW_NUMBER() OVER (ORDER BY Date) AS date_id,
        date,
        DATEPART(DAY, CONVERT(Date, date)) AS day,
        DATEPART(MONTH, CONVERT(Date,date)) AS month, 
        DATEPART(QUARTER, CONVERT(Date,date)) AS quarter,
        DATEPART(YEAR, CONVERT(Date,date)) AS  year,
        DATEPART(DAYOFYEAR,CONVERT(Date,date)) AS date_of_year,
        DATEPART(WEEKDAY,CONVERT(Date,date)) AS date_of_month
    FROM
        dbo.staging_payment
);

Go
SELECT
    TOP 10*
FROM 
    [dbo].[dim_payment];