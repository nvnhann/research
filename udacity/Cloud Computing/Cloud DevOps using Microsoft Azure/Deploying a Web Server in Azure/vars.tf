variable "prefix" {
  description = "The prefix which should be used for all resources"
  default = "Azuredevops"
}

variable "location" {
  description = "The Azure Region in which all resources"
  default = "westeurope"
}

variable "username" {
  description = "The VM users name."
  default = "udacity"
}

variable "password" {
  description = "The VM users password:"
    default= "Abbc@1234"
}

variable "environment"{
  description = "The environment should be used for all resources"
  default = "test"
}

variable "server_names"{
  type = list
  default = ["uat","int"]
}

variable "number_of_vms" {
  description = "The number of Virtual Machines to be deployed."
  type        = number
  default     = 2
  validation {
    condition     = var.number_of_vms >= 2 && var.number_of_vms <= 5
    error_message = "The number of VMs must be at least 2 and no more than 5."
  }
}

variable "packer_image" {
  description = "The ID of the image created by packer tool."
  default = "/subscriptions/630a1e98-7922-4c13-9488-39768dd9328d/resourceGroups/Azuredevops/providers/Microsoft.Compute/images/PackerImage"
}

variable "subscription" {
  description = "The subscription for which the resources"
  default = "/subscriptions/630a1e98-7922-4c13-9488-39768dd9328d"
}