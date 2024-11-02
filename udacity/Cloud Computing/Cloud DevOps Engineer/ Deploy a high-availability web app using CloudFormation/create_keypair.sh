#!/bin/bash

# Prompt for the key pair name
read -p "Enter the name for the new EC2 key pair: " key_name

# Specify the AWS profile
read -p "Enter your AWS profile name: " aws_profile

# Create the EC2 key pair using the specified profile
aws ec2 create-key-pair --key-name "$key_name" --query 'KeyMaterial' --output text --profile "$aws_profile" > "${key_name}.pem"

# Check if the key was created and saved successfully
if [ $? -eq 0 ]; then
    echo "Key pair '${key_name}' created successfully and saved as ${key_name}.pem"
    chmod 400 "${key_name}.pem"
else
    echo "Failed to create key pair"
fi
