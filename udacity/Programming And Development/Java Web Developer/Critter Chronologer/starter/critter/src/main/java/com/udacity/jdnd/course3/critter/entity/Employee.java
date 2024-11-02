package com.udacity.jdnd.course3.critter.entity;

import com.udacity.jdnd.course3.critter.user.EmployeeSkill;

import javax.persistence.*;
import java.time.DayOfWeek;
import java.util.Set;

@Table
@Entity
public class Employee {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private String name;

    @ElementCollection(targetClass = EmployeeSkill.class)
    @Enumerated(EnumType.STRING)
    @CollectionTable(name = "employee_skill")
    @Column(name = "skill")
    private Set<EmployeeSkill> skills;

    @ElementCollection(targetClass = DayOfWeek.class)
    @Enumerated(EnumType.STRING)
    @CollectionTable(name = "employee_day_available")
    @Column(name = "day_available")
    private Set<DayOfWeek> daysAvailable;

    /**
     * Constructor with all fields.
     *
     * @param id            the employee id
     * @param name          the employee name
     * @param skills        the set of skills the employee possesses
     * @param daysAvailable the set of days the employee is available
     */
    public Employee(Long id, String name, Set<EmployeeSkill> skills, Set<DayOfWeek> daysAvailable) {
        this.id = id;
        this.name = name;
        this.skills = skills;
        this.daysAvailable = daysAvailable;
    }

    /**
     * Default constructor.
     */
    public Employee() {
    }

    /**
     * Gets the employee id.
     *
     * @return the employee id
     */
    public Long getId() {
        return id;
    }

    /**
     * Sets the employee id.
     *
     * @param id the employee id
     */
    public void setId(Long id) {
        this.id = id;
    }

    /**
     * Gets the employee name.
     *
     * @return the employee name
     */
    public String getName() {
        return name;
    }

    /**
     * Sets the employee name.
     *
     * @param name the employee name
     */
    public void setName(String name) {
        this.name = name;
    }

    /**
     * Gets the set of employee skills.
     *
     * @return the set of employee skills
     */
    public Set<EmployeeSkill> getSkills() {
        return skills;
    }

    /**
     * Gets the set of days the employee is available.
     *
     * @return the set of days the employee is available
     */
    public Set<DayOfWeek> getDaysAvailable() {
        return daysAvailable;
    }

    /**
     * Sets the set of days the employee is available.
     *
     * @param daysAvailable the set of days the employee is available
     */
    public void setDaysAvailable(Set<DayOfWeek> daysAvailable) {
        this.daysAvailable = daysAvailable;
    }


}