package com.udacity.jwdnd.course1.cloudstorage.model;

public class Credential {

    /**
     * The Credential class represents a user's credential information.
     * It includes fields for credential ID, URL, username, encryption key,
     * password, and user ID.
     */
    private Integer credentialId;
    private String url;
    private String username;
    private String key;
    private String password;
    private Integer userId;

    /**
     * Constructs a Credential object without userId.
     *
     * @param credentialId the ID of the credential
     * @param url          the URL associated with the credential
     * @param username     the username associated with the credential
     * @param key          the encryption key for the credential
     * @param password     the password for the credential
     */
    public Credential(
            Integer credentialId,
            String url,
            String username,
            String key,
            String password) {

        this.credentialId = credentialId;
        this.url = url;
        this.username = username;
        this.key = key;
        this.password = password;
    }

    /**
     * Constructs a Credential object with all fields.
     *
     * @param credentialId the ID of the credential
     * @param url          the URL associated with the credential
     * @param username     the username associated with the credential
     * @param key          the encryption key for the credential
     * @param password     the password for the credential
     * @param userId       the ID of the user associated with the credential
     */
    public Credential(
            Integer credentialId,
            String url,
            String username,
            String key,
            String password,
            Integer userId) {

        this.credentialId = credentialId;
        this.url = url;
        this.username = username;
        this.key = key;
        this.password = password;
        this.userId = userId;
    }

    /**
     * Gets the ID of the credential.
     *
     * @return the credential ID
     */
    public Integer getCredentialId() {
        return credentialId;
    }

    /**
     * Sets the ID of the credential.
     *
     * @param credentialId the new credential ID
     */
    public void setCredentialId(Integer credentialId) {
        this.credentialId = credentialId;
    }

    /**
     * Gets the URL associated with the credential.
     *
     * @return the URL
     */
    public String getUrl() {
        return url;
    }

    /**
     * Sets the URL associated with the credential.
     *
     * @param url the new URL
     */
    public void setUrl(String url) {
        this.url = url;
    }

    /**
     * Gets the username associated with the credential.
     *
     * @return the username
     */
    public String getUsername() {
        return username;
    }

    /**
     * Sets the username associated with the credential.
     *
     * @param username the new username
     */
    public void setUsername(String username) {
        this.username = username;
    }

    /**
     * Gets the encryption key for the credential.
     *
     * @return the encryption key
     */
    public String getKey() {
        return key;
    }

    /**
     * Sets the encryption key for the credential.
     *
     * @param key the new encryption key
     */
    public void setKey(String key) {
        this.key = key;
    }

    /**
     * Gets the password for the credential.
     *
     * @return the password
     */
    public String getPassword() {
        return password;
    }

    /**
     * Sets the password for the credential.
     *
     * @param password the new password
     */
    public void setPassword(String password) {
        this.password = password;
    }

    /**
     * Gets the ID of the user associated with the credential.
     *
     * @return the user ID
     */
    public Integer getUserId() {
        return userId;
    }

    /**
     * Sets the ID of the user associated with the credential.
     *
     * @param userId the new user ID
     */
    public void setUserId(Integer userId) {
        this.userId = userId;
    }
}