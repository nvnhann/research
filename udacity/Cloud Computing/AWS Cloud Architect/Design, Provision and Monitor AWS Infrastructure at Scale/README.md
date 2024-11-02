Project Repository

```
├── Exercise_1
│   └── main.tf
├── Exercise_2
│   ├── greet_lambda.py
│   ├── main.tf
│   ├── outputs.tf
│   ├── output.zip
│   └── variables.tf
├── Task1
│   ├── Udacity_Diagram_1.pdf
│   └── Udacity_Diagram_2.pdf
├── Task2
│   ├── Increased_Cost_Estimate.csv
│   ├── Initial_Cost_Estimate.csv
│   └── Reduced_Cost_Estimate.csv
├── Task5
│   ├── Terraform_1_1.png
│   ├── Terraform_1_2.png
│   ├── Terraform_2_1.png
│   ├── Terraform_2_2.png
│   └── Terraform_2_3.png
└── Task6
    └── Terraform_destroyed.png
```

I can identify the changes made to the services in my infrastructure. Here's a breakdown:

Services with Reduced Costs:

Amazon RDS for MySQL: The monthly cost has decreased from 8261.345 USD to 4218.325 USD. The total 12 months cost is now 50619.90 USD. The quantity of RDS instances has been reduced from 4 to 2.

Services with Increased Costs:

VPN Connection: The monthly cost has increased from 73 USD to 146 USD. The total 12 months cost is now 1752.00 USD. The number of Site-to-Site VPN Connections has been doubled from 2 to 4.

No New Services or Removals:
There don't seem to be any new services added or removed from me infrastructure. The changes made involve adjusting the quantities and configurations of existing services, specifically the VPN Connection and Amazon RDS for MySQL.

# Project Repository

This repository contains the following files and directories:

## Exercise_1

- `main.tf`: Terraform configuration file for Exercise 1.

## Exercise_2

- `greet_lambda.py`: Python script for Exercise 2.
- `main.tf`: Terraform configuration file for Exercise 2.
- `outputs.tf`: Terraform configuration file defining output values.
- `output.zip`: Zip archive containing additional files for Exercise 2.
- `variables.tf`: Terraform configuration file defining input variables.

## Task1

- `Udacity_Diagram_1.pdf`: Diagram file for Task 1.
- `Udacity_Diagram_2.pdf`: Diagram file for Task 1.

## Task2

- `Increased_Cost_Estimate.csv`: Cost estimate file for Task 2.
- `Initial_Cost_Estimate.csv`: Cost estimate file for Task 2.
- `Reduced_Cost_Estimate.csv`: Cost estimate file for Task 2.

## Task5

- `Terraform_1_1.png`: Screenshot of EC2 instances for Task 5.
- `Terraform_1_2.png`: Screenshot of EC2 instances for Task 5.
- `Terraform_2_1.png`: Screenshot of EC2 instances for Task 5.
- `Terraform_2_2.png`: Screenshot of EC2 instances for Task 5.
- `Terraform_2_3.png`: Screenshot of EC2 instances for Task 5.

## Task6

- `Terraform_destroyed.png`: Screenshot of destroyed infrastructure for Task 6.
