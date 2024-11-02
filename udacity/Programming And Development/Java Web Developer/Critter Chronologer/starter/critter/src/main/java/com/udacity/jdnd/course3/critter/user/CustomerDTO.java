package com.udacity.jdnd.course3.critter.user;

import java.util.List;

/**
 * Represents the form that customer request and response data takes. Does not map
 * to the database directly.
 */
public class CustomerDTO {
    private long id;
    private String name;
    private String phoneNumber;
    private String notes;
    private List<Long> petIds;

    public CustomerDTO() {
    }

    public CustomerDTO(long id, String name, String phoneNumber, String notes, List<Long> petIds) {
        this.id = id;
        this.name = name;
        this.phoneNumber = phoneNumber;
        this.notes = notes;
        this.petIds = petIds;
    }

    /**
     * Gets the ID of the customer.
     *
     * @return the ID of the customer
     */
    public long getId() {
        return id;
    }

    /**
     * Sets the ID of the customer.
     *
     * @param id the ID to set
     */
    public void setId(long id) {
        this.id = id;
    }

    /**
     * Gets the name of the customer.
     *
     * @return the name of the customer
     */
    public String getName() {
        return name;
    }

    /**
     * Sets the name of the customer.
     *
     * @param name the name to set
     */
    public void setName(String name) {
        this.name = name;
    }

    /**
     * Gets the phone number of the customer.
     *
     * @return the phone number of the customer
     */
    public String getPhoneNumber() {
        return phoneNumber;
    }

    /**
     * Sets the phone number of the customer.
     *
     * @param phoneNumber the phone number to set
     */
    public void setPhoneNumber(String phoneNumber) {
        this.phoneNumber = phoneNumber;
    }

    /**
     * Gets the notes for the customer.
     *
     * @return the notes for the customer
     */
    public String getNotes() {
        return notes;
    }

    /**
     * Sets the notes for the customer.
     *
     * @param notes the notes to set
     */
    public void setNotes(String notes) {
        this.notes = notes;
    }

    /**
     * Gets the list of pet IDs associated with the customer.
     *
     * @return the list of pet IDs
     */
    public List<Long> getPetIds() {
        return petIds;
    }

    /**
     * Sets the list of pet IDs associated with the customer.
     *
     * @param petIds the list of pet IDs to set
     */
    public void setPetIds(List<Long> petIds) {
        this.petIds = petIds;
    }
}