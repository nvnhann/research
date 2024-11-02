package com.udacity.jdnd.course3.critter.schedule;

import com.udacity.jdnd.course3.critter.entity.Employee;
import com.udacity.jdnd.course3.critter.entity.Pet;
import com.udacity.jdnd.course3.critter.entity.Schedule;
import com.udacity.jdnd.course3.critter.repository.EmployeeRepository;
import com.udacity.jdnd.course3.critter.repository.PetRepository;
import com.udacity.jdnd.course3.critter.service.ScheduleService;
import org.springframework.beans.BeanUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.server.ResponseStatusException;

import java.util.List;
import java.util.stream.Collectors;

/**
 * Handles web requests related to Schedules.
 */
@RestController
@RequestMapping("/schedule")
public class ScheduleController {

    @Autowired
    ScheduleService scheduleService;
    @Autowired
    private EmployeeRepository employeeRepository;
    @Autowired
    private PetRepository petRepository;

    /**
     * Converts a Schedule entity to a ScheduleDTO.
     *
     * @param schedule the Schedule entity to convert
     * @return the converted ScheduleDTO
     */
    private ScheduleDTO convertScheduleToScheduleDTO(Schedule schedule) {
        List<Long> employeeIds = schedule.getEmployees().stream().map(Employee::getId).collect(Collectors.toList());
        List<Long> petIds = schedule.getPets().stream().map(Pet::getId).collect(Collectors.toList());
        return new ScheduleDTO(schedule.getId(), employeeIds, petIds, schedule.getDate(), schedule.getActivities());
    }

    /**
     * Creates and saves a new schedule.
     *
     * @param scheduleDTO the DTO containing schedule details
     * @return the saved ScheduleDTO
     */
    @PostMapping
    public ScheduleDTO createSchedule(@RequestBody ScheduleDTO scheduleDTO) {
        Schedule schedule = convertDTOToSchedule(scheduleDTO);

        Schedule savedSchedule = scheduleService.saveSchedule(schedule);

        return convertScheduleToDTO(savedSchedule);

    }

    public ScheduleDTO convertScheduleToDTO(Schedule schedule){
        ScheduleDTO scheduleDTO = new ScheduleDTO();
        BeanUtils.copyProperties(schedule, scheduleDTO);

        List<Long> listOfEmployeeIDs = schedule.getEmployees()
                .stream()
                .map(Employee::getId)
                .collect(Collectors.toList());

        List<Long> listOfPetIDs = schedule.getPets()
                .stream()
                .map(Pet::getId)
                .collect(Collectors.toList());

        scheduleDTO.setDate(schedule.getDate());
        scheduleDTO.setActivities(schedule.getActivities());
        scheduleDTO.setEmployeeIds(listOfEmployeeIDs);
        scheduleDTO.setPetIds(listOfPetIDs);

        return scheduleDTO;
    }

    /**
     * Retrieves all schedules.
     *
     * @return list of all ScheduleDTOs
     */
    @GetMapping
    public List<ScheduleDTO> getAllSchedules() {
        return scheduleService.getAllSchedules()
                .stream()
                .map(this::convertScheduleToScheduleDTO)
                .collect(Collectors.toList());
    }

    /**
     * Retrieves schedules for a specific pet.
     *
     * @param petId the ID of the pet
     * @return list of ScheduleDTOs for the pet
     */
    @GetMapping("/pet/{petId}")
    public List<ScheduleDTO> getScheduleForPet(@PathVariable long petId) {
        List<Schedule> schedules;
        try {
            schedules = scheduleService.getPetSchedule(petId);
        } catch (Exception exception) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "Pet schedule with id: " + petId + " not found", exception);
        }
        return schedules.stream().map(this::convertScheduleToScheduleDTO).collect(Collectors.toList());
    }

    /**
     * Retrieves schedules for a specific employee.
     *
     * @param employeeId the ID of the employee
     * @return list of ScheduleDTOs for the employee
     */
    @GetMapping("/employee/{employeeId}")
    public List<ScheduleDTO> getScheduleForEmployee(@PathVariable long employeeId) {
        List<Schedule> schedules;
        try {
            schedules = scheduleService.getEmployeeSchedule(employeeId);
        } catch (Exception exception) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "Employee schedule with employee id: " + employeeId + " not found", exception);
        }
        return schedules.stream().map(this::convertScheduleToScheduleDTO).collect(Collectors.toList());
    }

    /**
     * Retrieves schedules for a specific customer.
     *
     * @param customerId the ID of the customer
     * @return list of ScheduleDTOs for the customer
     */
    @GetMapping("/customer/{customerId}")
    public List<ScheduleDTO> getScheduleForCustomer(@PathVariable long customerId) {
        List<Schedule> schedules;
        try {
            schedules = scheduleService.getCustomerSchedule(customerId);
        } catch (Exception exception) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "Schedule with owner id " + customerId + " not found", exception);
        }
        return schedules.stream().map(this::convertScheduleToScheduleDTO).collect(Collectors.toList());
    }

    public Schedule convertDTOToSchedule(ScheduleDTO scheduleDTO){
        Schedule schedule = new Schedule();

        BeanUtils.copyProperties(scheduleDTO, schedule);

        schedule.setDate(scheduleDTO.getDate());
        schedule.setSkills(scheduleDTO.getActivities());
        schedule.setEmployees(employeeRepository.findAllById(scheduleDTO.getEmployeeIds()));
        schedule.setPets(petRepository.findAllById(scheduleDTO.getPetIds()));

        return schedule;
    }
}