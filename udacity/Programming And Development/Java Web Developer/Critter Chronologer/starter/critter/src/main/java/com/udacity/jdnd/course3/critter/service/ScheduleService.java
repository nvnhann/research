package com.udacity.jdnd.course3.critter.service;

import com.udacity.jdnd.course3.critter.entity.Customer;
import com.udacity.jdnd.course3.critter.entity.Employee;
import com.udacity.jdnd.course3.critter.entity.Pet;
import com.udacity.jdnd.course3.critter.entity.Schedule;
import com.udacity.jdnd.course3.critter.repository.CustomerRepository;
import com.udacity.jdnd.course3.critter.repository.EmployeeRepository;
import com.udacity.jdnd.course3.critter.repository.PetRepository;
import com.udacity.jdnd.course3.critter.repository.ScheduleRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import javax.transaction.Transactional;
import java.util.List;

@Service
@Transactional
public class ScheduleService {

    @Autowired
    private ScheduleRepository scheduleRepository;

    @Autowired
    private PetRepository petRepository;

    @Autowired
    private EmployeeRepository employeeRepository;

    @Autowired
    private CustomerRepository customerRepository;

    /**
     * Saves a new schedule with the specified employees and pets.
     *
     * @param schedule    the schedule to save
     * @return the saved schedule
     */
    public Schedule saveSchedule(Schedule schedule) {
        return scheduleRepository.save(schedule);
    }

    /**
     * Gets all schedules.
     *
     * @return the list of all schedules
     */
    public List<Schedule> getAllSchedules() {
        return scheduleRepository.findAll();
    }

    /**
     * Gets the schedule for a specific employee.
     *
     * @param employeeId the ID of the employee
     * @return the list of schedules for the employee
     */
    public List<Schedule> getEmployeeSchedule(Long employeeId) {
        Employee employee = employeeRepository.getOne(employeeId);
        return scheduleRepository.findByEmployee(employee);
    }

    /**
     * Gets the schedule for a specific pet.
     *
     * @param petId the ID of the pet
     * @return the list of schedules for the pet
     */
    public List<Schedule> getPetSchedule(Long petId) {
        Pet pet = petRepository.getOne(petId);
        return scheduleRepository.findByPets(pet);
    }

    /**
     * Gets the schedule for a specific customer.
     *
     * @param customerId the ID of the customer
     * @return the list of schedules for the customer's pets
     */
    public List<Schedule> getCustomerSchedule(Long customerId) {
        Customer customer = customerRepository.getOne(customerId);
        return scheduleRepository.findByPetsIn(customer.getPets());
    }
}