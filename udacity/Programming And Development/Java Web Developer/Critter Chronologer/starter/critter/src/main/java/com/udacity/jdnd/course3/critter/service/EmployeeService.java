package com.udacity.jdnd.course3.critter.service;

import com.udacity.jdnd.course3.critter.entity.Employee;
import com.udacity.jdnd.course3.critter.repository.EmployeeRepository;
import com.udacity.jdnd.course3.critter.user.EmployeeSkill;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import javax.transaction.Transactional;
import java.time.DayOfWeek;
import java.time.LocalDate;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

@Service
@Transactional
public class EmployeeService {

    @Autowired
    private EmployeeRepository employeeRepository;

    /**
     * Saves the given employee.
     *
     * @param employee the employee to save
     * @return the saved employee
     */
    public Employee saveEmployee(Employee employee) {
        return employeeRepository.save(employee);
    }

    /**
     * Retrieves a list of employees available on a given date with the required skills.
     *
     * @param date   the date to check availability
     * @param skills the skills required
     * @return a list of employees meeting the criteria
     */
    public List<Employee> getEmployeesByService(LocalDate date, Set<EmployeeSkill> skills) {
        return employeeRepository.findByDaysAvailable(date.getDayOfWeek()).stream()
                .filter(employee -> employee.getSkills().containsAll(skills))
                .collect(Collectors.toList());
    }

    /**
     * Retrieves an employee by their id.
     *
     * @param employeeId the id of the employee
     * @return the found employee
     */
    public Employee getEmployeeById(Long employeeId) {
        return employeeRepository.findById(employeeId).orElse(null);
    }

    /**
     * Sets the availability of an employee.
     *
     * @param days       the days of availability
     * @param employeeId the id of the employee
     */
    public void setEmployeeAvailability(Set<DayOfWeek> days, Long employeeId) {
        Employee employee = employeeRepository.findById(employeeId).orElseThrow(() ->
                new IllegalArgumentException("Employee not found with id: " + employeeId));
        employee.setDaysAvailable(days);
        employeeRepository.save(employee);
    }

}