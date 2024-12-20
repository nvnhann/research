resource "azurerm_virtual_network" "test" {
  name                 = format("%s-%s", var.application_type , var.resource_type)
  address_space        = var.address_space
  location             = var.location
  resource_group_name  = var.resource_group
}
resource "azurerm_subnet" "test" {
  name                    = format("%s-%s-sub", var.application_type, var.resource_type)
  resource_group_name     = var.resource_group
  virtual_network_name    = azurerm_virtual_network.test.name
  address_prefixes        = var.address_prefix_test
}
