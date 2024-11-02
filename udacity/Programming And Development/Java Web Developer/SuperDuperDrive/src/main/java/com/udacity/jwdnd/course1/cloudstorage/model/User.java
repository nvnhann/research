package com.udacity.jwdnd.course1.cloudstorage.model;

/**
 * User model representing a user entity.
 */
public class User {

    private Integer userid;
    private String username;
    private String salt;
    private String password;
    private String firstname;
    private String lastname;

    /**
     * Constructor to create a User instance with given details.
     *
     * @param username  the username
     * @param salt      the salt for password encryption
     * @param password  the encrypted password
     * @param firstname the first name
     * @param lastname  the last name
     */
    public User(String username, String salt, String password, String firstname, String lastname) {
        this.username = username;
        this.salt = salt;
        this.password = password;
        this.firstname = firstname;
        this.lastname = lastname;
    }

    /**
     * Constructor to create a User instance with given details including userId.
     *
     * @param userId    the user ID
     * @param username  the username
     * @param salt      the salt for password encryption
     * @param password  the encrypted password
     * @param firstname the first name
     * @param lastname  the last name
     */
    public User(Integer userId, String username, String salt, String password, String firstname, String lastname) {
        this.userid = userId;
        this.username = username;
        this.salt = salt;
        this.password = password;
        this.firstname = firstname;
        this.lastname = lastname;
    }

    /**
     * Gets the user ID.
     *
     * @return the user ID
     */
    public int getUserid() {
        return userid;
    }

    /**
     * Sets the user ID.
     *
     * @param userid the user ID to set
     */
    public void setUserid(int userid) {
        this.userid = userid;
    }

    /**
     * Gets the username.
     *
     * @return the username
     */
    public String getUsername() {
        return username;
    }

    /**
     * Sets the username.
     *
     * @param username the username to set
     */
    public void setUsername(String username) {
        this.username = username;
    }

    /**
     * Gets the salt for password encryption.
     *
     * @return the salt
     */
    public String getSalt() {
        return salt;
    }

    /**
     * Sets the salt for password encryption.
     *
     * @param salt the salt to set
     */
    public void setSalt(String salt) {
        this.salt = salt;
    }

    /**
     * Gets the encrypted password.
     *
     * @return the encrypted password
     */
    public String getPassword() {
        return password;
    }

    /**
     * Sets the encrypted password.
     *
     * @param password the encrypted password to set
     */
    public void setPassword(String password) {
        this.password = password;
    }

    /**
     * Gets the first name.
     *
     * @return the first name
     */
    public String getFirstname() {
        return firstname;
    }

    /**
     * Sets the first name.
     *
     * @param firstname the first name to set
     */
    public void setFirstname(String firstname) {
        this.firstname = firstname;
    }

    /**
     * Gets the last name.
     *
     * @return the last name
     */
    public String getLastname() {
        return lastname;
    }

    /**
     * Sets the last name.
     *
     * @param lastname the last name to set
     */
    public void setLastname(String lastname) {
        this.lastname = lastname;
    }
}