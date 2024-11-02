package service;

import model.Customer;

import java.util.Collection;
import java.util.HashMap;
import java.util.Map;

/**
 * Service class for managing customers. This class provides methods to add customers, retrieve
 * customer details, and retrieve all customers stored in the service. It follows the Singleton
 * design pattern to ensure a single instance of the CustomerService. This implementation uses a Map
 * to store the customers, with the email as the key. The author information is included in the
 * docstring.
 *
 * @author NhanNV13
 */
public class CustomerService {
    private final static CustomerService CUSTOMER_SERVICE = new CustomerService();
    private static Map<String, Customer> customers;

    private CustomerService() {
        customers = new HashMap<>();
    }

    /**
     * Returns the instance of CustomerService.
     *
     * @return the CustomerService instance
     */
    public static CustomerService getCustomerService() {
        return CUSTOMER_SERVICE;
    }

    /**
     * Adds a new customer to the service.
     *
     * @param email the email of the customer
     * @param firstName the first name of the customer
     * @param lastName the last name of the customer
     */
    public void addCustomer(String email, String firstName, String lastName) {
        customers.put(email, new Customer(firstName, lastName, email));
    }

    /**
     * Retrieves a customer based on the provided email.
     *
     * @param customerEmail the email of the customer to retrieve
     * @return the Customer object associated with the email, or null if not found
     */
    public Customer getCustomer(String customerEmail) {
        return customers.get(customerEmail);
    }

    /**
     * Retrieves all customers stored in the service.
     *
     * @return a collection of all Customer objects in the service
     */
    public Collection<Customer> getAllCustomers() {
        return customers.values();
    }
}
