package com.udacity.jdnd.course3.critter.entity;

import org.hibernate.annotations.Nationalized;

import javax.persistence.*;
import java.util.List;

/**
 * Represents a customer entity with details such as name, phone number, and associated pets.
 */
@Entity
@Table(name = "customer")
public class Customer {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Nationalized
    private String name;

    private String phoneNumber;
    private String notes;

    @OneToMany(mappedBy = "customer", cascade = CascadeType.ALL, orphanRemoval = true)
    private List<Pet> pets;

    /**
     * Constructs a Customer instance with the specified parameters.
     *
     * @param id          the ID of the customer
     * @param name        the name of the customer
     * @param phoneNumber the phone number of the customer
     * @param notes       any additional notes about the customer
     */
    public Customer(Long id, String name, String phoneNumber, String notes) {
        this.id = id;
        this.name = name;
        this.phoneNumber = phoneNumber;
        this.notes = notes;
    }

    /**
     * Default constructor for creating an empty Customer instance.
     */
    public Customer() {
        // Default constructor
    }

    // Getter and setter methods

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getName() {
        return name;
    }

    public String getPhoneNumber() {
        return phoneNumber;
    }


    public String getNotes() {
        return notes;
    }


    public List<Pet> getPets() {
        return pets;
    }

    public void setPets(List<Pet> pets) {
        this.pets = pets;
    }
}