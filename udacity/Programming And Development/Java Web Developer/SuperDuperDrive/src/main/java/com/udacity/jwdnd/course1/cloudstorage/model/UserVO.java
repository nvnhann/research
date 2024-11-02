package com.udacity.jwdnd.course1.cloudstorage.model;

/**
 * A class representing a user in the system.
 */

public class UserVO {

    private String username;
    private String password;
    private String firstName;
    private String lastName;

    /**
     * Default constructor.
     */
    public UserVO() {
    }

    /**
     * Parameterized constructor.
     *
     * @param username  the username of the user.
     * @param password  the password of the user.
     * @param firstName the first name of the user.
     * @param lastName  the last name of the user.
     */
    public UserVO(String username, String password, String firstName, String lastName) {
        this.username = username;
        this.password = password;
        this.firstName = firstName;
        this.lastName = lastName;
    }

    /**
     * Gets the username.
     *
     * @return the username of the user.
     */
    public String getUsername() {
        return username;
    }

    /**
     * Sets the username.
     *
     * @param username the username to set.
     */
    public void setUsername(String username) {
        this.username = username;
    }

    /**
     * Gets the password.
     *
     * @return the password of the user.
     */
    public String getPassword() {
        return password;
    }

    /**
     * Sets the password.
     *
     * @param password the password to set.
     */
    public void setPassword(String password) {
        this.password = password;
    }

    /**
     * Gets the first name.
     *
     * @return the first name of the user.
     */
    public String getFirstName() {
        return firstName;
    }

    /**
     * Sets the first name.
     *
     * @param firstName the first name to set.
     */
    public void setFirstName(String firstName) {
        this.firstName = firstName;
    }

    /**
     * Gets the last name.
     *
     * @return the last name of the user.
     */
    public String getLastName() {
        return lastName;
    }

    /**
     * Sets the last name.
     *
     * @param lastName the last name to set.
     */
    public void setLastName(String lastName) {
        this.lastName = lastName;
    }

    /**
     * Returns a string representation of the UserVO object.
     *
     * @return a string representation of the UserVO object.
     */
    @Override
    public String toString() {
        return "UserVO{" +
                "username='" + username + '\'' +
                ", password='" + password + '\'' +
                ", firstName='" + firstName + '\'' +
                ", lastName='" + lastName + '\'' +
                '}';
    }
}