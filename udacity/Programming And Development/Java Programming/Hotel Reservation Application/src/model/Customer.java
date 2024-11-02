package model;

/**
 * Represents a customer. Each customer has a first name, last name, and email address. This class
 * provides methods to validate the email address and retrieve customer information.
 *
 * @author NhanNV13
 */
public class Customer {
    private final String firstName;
    private final String lastName;
    private final String email;

    /**
     * Constructs a Customer object with the specified first name, last name, and email address. The
     * email address is validated during object creation.
     *
     * @param firstName the first name of the customer
     * @param lastName the last name of the customer
     * @param email the email address of the customer
     * @throws IllegalArgumentException if an invalid email address is provided
     */
    public Customer(String firstName, String lastName, String email) {
        this.validateEmail(email);
        this.firstName = firstName;
        this.lastName = lastName;
        this.email = email;
    }

    /**
     * Returns the email address of the customer.
     *
     * @return the email address
     */
    public String getEmail() {
        return this.email;
    }

    public String getFirstName(){
        return this.firstName;
    }

    public String getLastName(){
        return this.lastName;
    }

    /**
     * Validates the email address format using a regular expression.
     *
     * @param email the email address to be validated
     * @throws IllegalArgumentException if an invalid email address is provided
     */
    private void validateEmail(String email) {
        String regex = "^[\\w-_\\.+]*[\\w-_\\.]\\@([\\w]+\\.)+[\\w]+[\\w]$";
        if (!email.matches(regex)) {
            throw new IllegalArgumentException("Invalid Email address!");
        }
    }


    @Override
    public String toString() {
        return this.firstName + " __ " + this.lastName + " __ " + this.email;
    }
}
