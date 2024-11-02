package com.udacity.jdnd.course3.critter.user;

import com.udacity.jdnd.course3.critter.entity.Customer;
import com.udacity.jdnd.course3.critter.entity.Employee;
import com.udacity.jdnd.course3.critter.entity.Pet;
import com.udacity.jdnd.course3.critter.service.CustomerService;
import com.udacity.jdnd.course3.critter.service.EmployeeService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.server.ResponseStatusException;

import java.time.DayOfWeek;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * Handles web requests related to Users.
 * <p>
 * Includes requests for both customers and employees. Splitting this into separate user and customer controllers
 * would be fine too, though that is not part of the required scope for this class.
 */
@RestController
@RequestMapping("/user")
public class UserController {

    @Autowired
    private CustomerService customerService;

    @Autowired
    private EmployeeService employeeService;

    /**
     * Converts a Customer entity to a CustomerDTO.
     *
     * @param customer the Customer entity
     * @return the corresponding CustomerDTO
     */
    private CustomerDTO convertCustomerToCustomerDTO(Customer customer) {
        List<Long> petIds = customer.getPets().stream().map(Pet::getId).collect(Collectors.toList());
        return new CustomerDTO(customer.getId(), customer.getName(), customer.getPhoneNumber(), customer.getNotes(), petIds);
    }

    /**
     * Converts an Employee entity to an EmployeeDTO.
     *
     * @param employee the Employee entity
     * @return the corresponding EmployeeDTO
     */
    private EmployeeDTO convertEmployeeToEmployeeDTO(Employee employee) {
        return new EmployeeDTO(employee.getId(), employee.getName(), employee.getSkills(), employee.getDaysAvailable());
    }

    /**
     * Saves a new customer.
     *
     * @param customerDTO the customer details
     * @return the saved CustomerDTO
     */
    @PostMapping("/customer")
    public CustomerDTO saveCustomer(@RequestBody CustomerDTO customerDTO) {
        Customer customer = new Customer(customerDTO.getId(), customerDTO.getName(), customerDTO.getPhoneNumber(), customerDTO.getNotes());
        List<Long> petIds = customerDTO.getPetIds();

        try {
            return convertCustomerToCustomerDTO(customerService.saveCustomer(customer, petIds));
        } catch (Exception exception) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "Customer could not be saved", exception);
        }
    }

    /**
     * Retrieves all customers.
     *
     * @return a list of CustomerDTOs
     */
    @GetMapping("/customer")
    public List<CustomerDTO> getAllCustomers() {
        List<Customer> customers = customerService.getAllCustomers();
        return customers.stream().map(this::convertCustomerToCustomerDTO).collect(Collectors.toList());
    }

    /**
     * Retrieves a customer by pet ID.
     *
     * @param petId the pet ID
     * @return the corresponding CustomerDTO
     */
    @GetMapping("/customer/pet/{petId}")
    public CustomerDTO getOwnerByPet(@PathVariable long petId) {
        try {
            return convertCustomerToCustomerDTO(customerService.getCustomerByPetId(petId));
        } catch (Exception exception) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "Owner pet with id: " + petId + " not found", exception);
        }
    }

    /**
     * Saves a new employee.
     *
     * @param employeeDTO the employee details
     * @return the saved EmployeeDTO
     */
    @PostMapping("/employee")
    public EmployeeDTO saveEmployee(@RequestBody EmployeeDTO employeeDTO) {
        Employee employee = new Employee(employeeDTO.getId(), employeeDTO.getName(), employeeDTO.getSkills(), employeeDTO.getDaysAvailable());

        try {
            return convertEmployeeToEmployeeDTO(employeeService.saveEmployee(employee));
        } catch (Exception exception) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "Employee could not be saved", exception);
        }
    }

    /**
     * Retrieves an employee by ID.
     *
     * @param employeeId the employee ID
     * @return the corresponding EmployeeDTO
     */
    @GetMapping("/employee/{employeeId}")
    public EmployeeDTO getEmployee(@PathVariable long employeeId) {
        try {
            return convertEmployeeToEmployeeDTO(employeeService.getEmployeeById(employeeId));
        } catch (Exception exception) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "Employee with id: " + employeeId + " not found", exception);
        }
    }

    /**
     * Sets the availability of an employee.
     *
     * @param daysAvailable the days the employee is available
     * @param employeeId    the employee ID
     */
    @PutMapping("/employee/{employeeId}")
    public void setAvailability(@RequestBody Set<DayOfWeek> daysAvailable, @PathVariable long employeeId) {
        try {
            employeeService.setEmployeeAvailability(daysAvailable, employeeId);
        } catch (Exception exception) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "Employee with id: " + employeeId + " not found", exception);
        }
    }

    /**
     * Finds employees available for a specific service.
     *
     * @param employeeDTO the employee request details
     * @return a list of EmployeeDTOs
     */
    @GetMapping("/employee/availability")
    public List<EmployeeDTO> findEmployeesForService(@RequestBody EmployeeRequestDTO employeeDTO) {
        List<Employee> employees = employeeService.getEmployeesByService(employeeDTO.getDate(), employeeDTO.getSkills());
        return employees.stream().map(this::convertEmployeeToEmployeeDTO).collect(Collectors.toList());
    }
}