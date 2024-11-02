package com.udacity.jdnd.course3.critter.user;

import java.time.LocalDate;
import java.util.Set;

/**
 * Represents a request to find available employees by skills. Does not map
 * to the database directly.
 */
public class EmployeeRequestDTO {
    private Set<EmployeeSkill> skills;
    private LocalDate date;

    /**
     * Gets the required skills for this request.
     *
     * @return the set of required {@link EmployeeSkill}
     */
    public Set<EmployeeSkill> getSkills() {
        return skills;
    }

    /**
     * Sets the required skills for this request.
     *
     * @param skills the set of required {@link EmployeeSkill}
     */
    public void setSkills(Set<EmployeeSkill> skills) {
        this.skills = skills;
    }

    /**
     * Gets the preferred date for the request.
     *
     * @return the {@link LocalDate} of the request
     */
    public LocalDate getDate() {
        return date;
    }

    /**
     * Sets the preferred date for the request.
     *
     * @param date the {@link LocalDate} of the request
     */
    public void setDate(LocalDate date) {
        this.date = date;
    }
}