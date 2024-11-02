package com.udacity.jdnd.course3.critter.entity;

import com.udacity.jdnd.course3.critter.user.EmployeeSkill;

import javax.persistence.*;
import java.time.LocalDate;
import java.util.List;
import java.util.Set;

/**
 * Represents a schedule entity which includes employees, pets, date, and activities.
 */
@Table
@Entity
public class Schedule {

    @Id
    @GeneratedValue
    private Long id;

    @ManyToMany(targetEntity = Employee.class)
    private List<Employee> employee;

    @ManyToMany(targetEntity = Pet.class)
    private List<Pet> pets;

    private LocalDate date;

    @ElementCollection
    private Set<EmployeeSkill> activities;

    /**
     * Constructs a Schedule instance with the specified date and activities.
     *
     * @param date       the date of the schedule
     * @param activities the set of scheduled activities
     */
    public Schedule(LocalDate date, Set<EmployeeSkill> activities) {
        this.date = date;
        this.activities = activities;
    }

    /**
     * Default constructor for creating an empty Schedule instance.
     */
    public Schedule() {
    }

    /**
     * Gets the schedule id.
     *
     * @return the schedule id
     */
    public Long getId() {
        return id;
    }

    /**
     * Sets the schedule id.
     *
     * @param id the schedule id
     */
    public void setId(Long id) {
        this.id = id;
    }

    /**
     * Gets the list of employees.
     *
     * @return the list of employees
     */
    public List<Employee> getEmployees() {
        return employee;
    }

    /**
     * Sets the list of employees.
     *
     * @param employees the list of employees
     */
    public void setEmployees(List<Employee> employees) {
        this.employee = employees;
    }

    /**
     * Gets the list of pets.
     *
     * @return the list of pets
     */
    public List<Pet> getPets() {
        return pets;
    }

    /**
     * Sets the list of pets.
     *
     * @param pets the list of pets
     */
    public void setPets(List<Pet> pets) {
        this.pets = pets;
    }

    /**
     * Gets the date of the schedule.
     *
     * @return the schedule date
     */
    public LocalDate getDate() {
        return date;
    }


    /**
     * Gets the set of scheduled activities.
     *
     * @return the set of scheduled activities
     */
    public Set<EmployeeSkill> getActivities() {
        return activities;
    }


    /**
     * Sets the date of the schedule.
     *
     * @param date the schedule date
     */
    public void setDate(LocalDate date) {
        this.date = date;
    }

    /**
     * Sets the set of scheduled activities.
     *
     * @param activities the set of scheduled activities
     */
    public void setSkills(Set<EmployeeSkill> activities) {
        this.activities = activities;
    }

}