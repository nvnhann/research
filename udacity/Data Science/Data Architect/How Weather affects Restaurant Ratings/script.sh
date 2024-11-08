./snow/snowsql -a dyppjgo-mt25954 -u nvnhan
USE udacity.STAGING;

# json
PUT file:///home/nvnhan/udacity/nd038/prj2/yelp_academic_dataset_business.json @json_stage auto_compress=true parallel=4;
PUT file:///home/nvnhan/udacity/nd038/prj2/yelp_academic_dataset_checkin.json @json_stage auto_compress=true parallel=4;
PUT file:///home/nvnhan/udacity/nd038/prj2/yelp_academic_dataset_review.json @json_stage auto_compress=true parallel=4;
PUT file:///home/nvnhan/udacity/nd038/p rj2/yelp_academic_dataset_tip.json @json_stage auto_compress=true parallel=4;
PUT file:///home/nvnhan/udacity/nd038/prj2/yelp_academic_dataset_user.json @json_stage auto_compress=true parallel=4;
PUT file:///home/nvnhan/udacity/nd038/prj2/yelp_academic_dataset_covid_features.json @json_stage auto_compress=true parallel=4;

copy into "YELP_ACADEMIC_DATASET_BUSINESS" from @json_stage/yelp_academic_dataset_business.json.gz file_format=jsonformat on_error='skip_file';
copy into "YELP_ACADEMIC_DATASET_CHECKIN" from @json_stage/yelp_academic_dataset_checkin.json.gz file_format=jsonformat on_error='skip_file';
copy into "YELP_ACADEMIC_DATASET_COVID_FEATURES" from @json_stage/yelp_academic_dataset_covid_features.json.gz file_format=jsonformat on_error='skip_file';
copy into "YELP_ACADEMIC_DATASET_REVIEW" from @json_stage/yelp_academic_dataset_review.json.gz file_format=jsonformat on_error='skip_file';
copy into "YELP_ACADEMIC_DATASET_TIP" from @json_stage/yelp_academic_dataset_tip.json.gz file_format=jsonformat on_error='skip_file';
copy into "YELP_ACADEMIC_DATASET_USER" from @json_stage/yelp_academic_dataset_user.json.gz file_format=jsonformat on_error='skip_file';

# csv

PUT file:///home/nvnhan/udacity/nd038/prj2/USW00023169-temperature-degreeF.csv @csv_stage auto_compress=true parallel=4;
PUT file:///home/nvnhan/udacity/nd038/prj2/USW00023169-LAS_VEGAS_MCCARRAN_INTL_AP-precipitation-inch.csv @csv_stage auto_compress=true parallel=4;

copy into "temperature"("date", "min", "max", "normal_min", "normal_max")from (select to_date(t.$1, 'yyyymmdd'), t.$2, t.$3, t.$4, t.$5 from @csv_stage/USW00023169-temperature-degreeF.csv t) file_format=csvformat ON_ERROR = 'continue';
copy into "precipitation"("date", "precipitation", "precipitation_normal") from (select to_date(t.$1, 'yyyymmdd'), t.$2, t.$3 from @csv_stage/USW00023169-LAS_VEGAS_MCCARRAN_INTL_AP-precipitation-inch.csv.gz t) file_format=csvformat ON_ERROR = 'continue';

