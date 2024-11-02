-- CREATE dim_payment TABLE
IF OBJECT_ID('dbo.dim_payment') IS NOT NULL
BEGIN
    DROP EXTERNAL TABLE dbo.dim_payment
END

CREATE EXTERNAL TABLE [dbo].[dim_payment] WITH(
    LOCATION = 'dim_payment',
	DATA_SOURCE = [AzureDataResource],
	FILE_FORMAT = [SynapseDelimitedTextFormat]
) AS (
    SELECT
        sp.payment_id,    
        sp.rider_id,
	    sp.amount,
        sp.date
FROM 
	staging_payment sp
    JOIN staging_rider sr ON sr.Rider_Id = sp.rider_id;
);

Go
SELECT
    TOP 10*
FROM 
    [dbo].[fact_payment];