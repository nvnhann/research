# Specify AWS as the cloud provider
provider "aws" {
  region = "us-east-1"  # Replace with your desired region
}

# Provision EC2 Instances
resource "aws_instance" "udacity_t2" {
  count         = 4
  ami           = "ami-0c94855ba95c71c99"  # Replace with the desired AMI ID
  instance_type = "t2.micro"
  tags = {
    Name = "Udacity T2"
  }
}

# resource "aws_instance" "udacity_m4" {
#   count         = 2
#   ami           = "ami-0c94855ba95c71c99"  # Replace with the desired AMI ID
#   instance_type = "m4.large"
#   tags = {
#     Name = "Udacity M4"
#   }
# }
