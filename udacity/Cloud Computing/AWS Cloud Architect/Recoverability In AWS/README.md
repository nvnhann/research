# Availability and Resilience Infrastructure Setup

This project demonstrates the creation of a highly available and resilient infrastructure in AWS. It focuses on database redundancy, website versioning, and recovery. The project showcases the setup, configuration, and monitoring of the infrastructure components.

## Folder Structure

- `estimates.txt`
- `log_rr_affter_promotion.txt`
- `log_rr_before_promotion.txt`
- `s3` folder:
  - `fall.jpg`
  - `index.html`
  - `spring.jpg`
  - `summer.jpg`
  - `winter.jpg`
- `Screenshots` folder:
  - `monitoring_connections.png`
  - `monitoring_replication.png`
  - `primaryDB_config.png`
  - `primaryDB_subnetgroup.png`
  - `primary_subnet_routing.png`
  - `primary_Vpc.png`
  - `primaryVPC_subnets.png`
  - `rr_after_promotion_1.png`
  - `rr_after_promotion.png`
  - `rr_before_promotion_1.png`
  - `rr_before_promotion.png`
  - `s3_delete_marker.png`
  - `s3_delete_revert.png`
  - `s3_deletion.png`
  - `s3_original.png`
  - `s3_session.png`
  - `s3_session_revert.png`
  - `secondaryDB_config.png`
  - `secondaryDB_subnetgroup.png`
  - `secondary_subnet_routing.png`
  - `secondary_vpc.png`
  - `secondary_Vpc.png`
  - `secondaryVPC_subnets.png`


- `estimates.txt`: Contains estimates or calculations related to the project's requirements or resource usage.

- `log_rr_after_promotion.txt`: Log file documenting student interactions with the read-replica database after promotion.

- `log_rr_before_promotion.txt`: Log file documenting student interactions with the read-replica database before promotion.

- `s3` folder: Contains various image files and an HTML file for the website versioning and recovery demonstration.

- `Screenshots` folder: Contains screenshots of different configurations and settings used in the project.

## Project Components

### Database Redundancy

- Primary Database:
  - Configuration screenshots: `Screenshots/primaryDB_config.png`, `Screenshots/primaryDB_subnetgroup.png`.
  - Route table configuration screenshot: `Screenshots/primary_subnet_routing.png`.
  - VPC configuration screenshot: `Screenshots/primary_Vpc.png`.
  - Subnet configuration screenshot: `Screenshots/primaryVPC_subnets.png`.

- Read-Replica Database:
  - Configuration screenshots: `Screenshots/secondaryDB_config.png`, `Screenshots/secondaryDB_subnetgroup.png`.
  - Route table configuration screenshot: `Screenshots/secondary_subnet_routing.png`.
  - VPC configuration screenshots: `Screenshots/secondary_vpc.png`, `Screenshots/secondary_Vpc.png`.
  - Subnet configuration screenshot: `Screenshots/secondaryVPC_subnets.png`.

### Monitoring and Resilience

- Monitoring:
  - Database connections metric screenshot: `Screenshots/monitoring_connections.png`.
  - Database replication screenshot: `Screenshots/monitoring_replication.png`.

- Resilient Database:
  - Read-replica before promotion screenshots: `Screenshots/rr_before_promotion_1.png`, `Screenshots/rr_before_promotion.png`.
  - Read-replica after promotion screenshots: `Screenshots/rr_after_promotion_1.png`, `Screenshots/rr_after_promotion.png`.
  - Log of student interactions with the standby database: `log_rr_affter_promotion.txt`.

### Website Versioning and Recovery

- S3 folder screenshots:
  - Original website screenshot: `Screenshots/s3_original.png`.
  - Website with different seasons screenshot: `Screenshots/s3_session.png`.
  - Deletion of a version screenshot: `Screenshots/s3_deletion.png`.
  - Reversion of session to a specific version screenshot: `Screenshots/s3_session_revert.png`.
  - Deletion marker screenshot: `Screenshots/s3_delete_marker.png`.
  - Reversal of deletion marker screenshot: `Screenshots/s3_delete_revert.png`.

## Usage

1. Set up the primary and read-replica databases as described in the configuration screenshots.

2. Monitor the database using the provided monitoring screenshots.

3. Promote the read-replica database and observe the changes.

4. Review the logs of student interactions with the standby database after promotion.

5. Explore the website versioning and recovery process using the screenshots in the S3 folder.

## Website

Access the project's website at [http://uacity-pro1.s3-website-us-east-1.amazonaws.com/](http://uacity-pro1.s3-website-us-east-1.amazonaws.com/).


## License

This project is licensed under the [MIT License](LICENSE).

Feel free to modify the structure and content according to your project's specific requirements and details.