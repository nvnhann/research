package com.udacity.jdnd.course3.critter.user;

import java.time.DayOfWeek;
import java.util.Set;

/**
 * Represents the form that employee request and response data takes. Does not map
 * to the database directly.
 */
public class EmployeeDTO {
    private long id;
    private String name;
    private Set<EmployeeSkill> skills;
    private Set<DayOfWeek> daysAvailable;

    /**
     * Constructs an EmployeeDTO with the given id, name, skills, and days available.
     *
     * @param id            the employee's id
     * @param name          the employee's name
     * @param skills        the employee's skills
     * @param daysAvailable the days the employee is available
     */
    public EmployeeDTO(long id, String name, Set<EmployeeSkill> skills, Set<DayOfWeek> daysAvailable) {
        this.id = id;
        this.name = name;
        this.skills = skills;
        this.daysAvailable = daysAvailable;
    }

    /**
     * Default no-arg constructor.
     */
    public EmployeeDTO() {
    }

    /**
     * Gets the employee's id.
     *
     * @return the id
     */
    public long getId() {
        return id;
    }

    /**
     * Sets the employee's id.
     *
     * @param id the id to set
     */
    public void setId(long id) {
        this.id = id;
    }

    /**
     * Gets the employee's name.
     *
     * @return the name
     */
    public String getName() {
        return name;
    }

    /**
     * Sets the employee's name.
     *
     * @param name the name to set
     */
    public void setName(String name) {
        this.name = name;
    }

    /**
     * Gets the employee's skills.
     *
     * @return the skills
     */
    public Set<EmployeeSkill> getSkills() {
        return skills;
    }

    /**
     * Sets the employee's skills.
     *
     * @param skills the skills to set
     */
    public void setSkills(Set<EmployeeSkill> skills) {
        this.skills = skills;
    }

    /**
     * Gets the days the employee is available.
     *
     * @return the days available
     */
    public Set<DayOfWeek> getDaysAvailable() {
        return daysAvailable;
    }

    /**
     * Sets the days the employee is available.
     *
     * @param daysAvailable the days available to set
     */
    public void setDaysAvailable(Set<DayOfWeek> daysAvailable) {
        this.daysAvailable = daysAvailable;
    }
}