package com.udacity.jdnd.course3.critter.entity;

import com.udacity.jdnd.course3.critter.pet.PetType;
import org.hibernate.annotations.Nationalized;

import javax.persistence.*;
import java.time.LocalDate;

/**
 * Represents a pet entity with relevant details such as type, name, birth date, and associated customer.
 */
@Table(name = "pet")
@Entity
public class Pet {
    @Id
    @GeneratedValue
    private Long id;

    private PetType type;
    @Nationalized
    private String name;
    @ManyToOne(targetEntity = Customer.class, optional = false)
    private Customer customer;
    private LocalDate birthDate;
    private String notes;

    /**
     * Constructs a Pet instance with the specified parameters.
     *
     * @param type      the type of the pet
     * @param name      the name of the pet
     * @param birthDate the birth date of the pet
     * @param notes     any additional notes about the pet
     */
    public Pet(PetType type, String name, LocalDate birthDate, String notes) {
        this.type = type;
        this.name = name;
        this.birthDate = birthDate;
        this.notes = notes;
    }

    /**
     * Default constructor for creating an empty Pet instance.
     */
    public Pet() {
    }

    // Getter and setter methods

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public PetType getType() {
        return type;
    }


    public String getName() {
        return name;
    }

    public Customer getCustomer() {
        return customer;
    }

    public void setCustomer(Customer customer) {
        this.customer = customer;
    }

    public LocalDate getBirthDate() {
        return birthDate;
    }

    public String getNotes() {
        return notes;
    }
}