package com.udacity.jdnd.course3.critter.schedule;

import com.udacity.jdnd.course3.critter.user.EmployeeSkill;

import java.time.LocalDate;
import java.util.List;
import java.util.Set;

/**
 * Represents the form that schedule request and response data takes. Does not map
 * to the database directly.
 */
public class ScheduleDTO {
    private long id;
    private List<Long> employeeIds;
    private List<Long> petIds;
    private LocalDate date;
    private Set<EmployeeSkill> activities;

    public ScheduleDTO(long id, List<Long> employeeIds, List<Long> petIds, LocalDate date, Set<EmployeeSkill> activities) {
        this.id = id;
        this.employeeIds = employeeIds;
        this.petIds = petIds;
        this.date = date;
        this.activities = activities;
    }

    public ScheduleDTO() {
    }

    /**
     * Gets the ID of the schedule.
     *
     * @return the ID of the schedule
     */
    public long getId() {
        return id;
    }

    /**
     * Sets the ID of the schedule.
     *
     * @param id the ID to set
     */
    public void setId(long id) {
        this.id = id;
    }

    /**
     * Gets the list of employee IDs.
     *
     * @return the list of employee IDs
     */
    public List<Long> getEmployeeIds() {
        return employeeIds;
    }

    /**
     * Sets the list of employee IDs.
     *
     * @param employeeIds the list of employee IDs to set
     */
    public void setEmployeeIds(List<Long> employeeIds) {
        this.employeeIds = employeeIds;
    }

    /**
     * Gets the list of pet IDs.
     *
     * @return the list of pet IDs
     */
    public List<Long> getPetIds() {
        return petIds;
    }

    /**
     * Sets the list of pet IDs.
     *
     * @param petIds the list of pet IDs to set
     */
    public void setPetIds(List<Long> petIds) {
        this.petIds = petIds;
    }

    /**
     * Gets the date of the schedule.
     *
     * @return the date of the schedule
     */
    public LocalDate getDate() {
        return date;
    }

    /**
     * Sets the date of the schedule.
     *
     * @param date the date to set
     */
    public void setDate(LocalDate date) {
        this.date = date;
    }

    /**
     * Gets the set of activities in the schedule.
     *
     * @return the set of activities
     */
    public Set<EmployeeSkill> getActivities() {
        return activities;
    }

    /**
     * Sets the set of activities in the schedule.
     *
     * @param activities the set of activities to set
     */
    public void setActivities(Set<EmployeeSkill> activities) {
        this.activities = activities;
    }
}