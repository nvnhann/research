package com.udacity.jdnd.course3.critter.service;

import com.udacity.jdnd.course3.critter.entity.Customer;
import com.udacity.jdnd.course3.critter.entity.Pet;
import com.udacity.jdnd.course3.critter.repository.CustomerRepository;
import com.udacity.jdnd.course3.critter.repository.PetRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import javax.transaction.Transactional;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

@Service
@Transactional
public class CustomerService {
    @Autowired
    private PetRepository petRepository;
    @Autowired
    private CustomerRepository customerRepository;

    /**
     * Saves the customer entity with associated pets.
     *
     * @param customer the customer entity to be saved.
     * @param petIds   the list of pet IDs to associate with the customer.
     * @return the saved Customer entity.
     */
    public Customer saveCustomer(Customer customer, List<Long> petIds) {
        List<Pet> customerPets = new ArrayList<>();
        if (petIds != null && !petIds.isEmpty()) {
            customerPets = petIds.stream()
                    .map(petRepository::getOne)
                    .collect(Collectors.toList());
        }
        customer.setPets(customerPets);
        return customerRepository.save(customer);
    }

    /**
     * Retrieves the customer entity associated with the given pet ID.
     *
     * @param petId the pet ID.
     * @return the Customer entity associated with the pet ID.
     */
    public Customer getCustomerByPetId(Long petId) {
        return petRepository.getOne(petId).getCustomer();
    }

    /**
     * Retrieves all customers.
     *
     * @return the list of all Customer entities.
     */
    public List<Customer> getAllCustomers() {
        return customerRepository.findAll();
    }
}