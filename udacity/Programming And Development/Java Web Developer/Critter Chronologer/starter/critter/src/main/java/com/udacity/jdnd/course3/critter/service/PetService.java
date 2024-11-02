package com.udacity.jdnd.course3.critter.service;

import com.udacity.jdnd.course3.critter.entity.Customer;
import com.udacity.jdnd.course3.critter.entity.Pet;
import com.udacity.jdnd.course3.critter.repository.CustomerRepository;
import com.udacity.jdnd.course3.critter.repository.PetRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import javax.transaction.Transactional;
import java.util.List;

@Service
@Transactional
public class PetService {
    @Autowired
    private CustomerRepository customerRepository;

    @Autowired
    private PetRepository petRepository;

    /**
     * Saves a pet entity associated with a specified customer.
     *
     * @param pet        the pet entity to save
     * @param customerId the ID of the customer to associate with the pet
     * @return the saved pet entity
     */
    public Pet savePet(Pet pet, Long customerId) {
        Customer customer = customerRepository.getOne(customerId);
        pet.setCustomer(customer);
        pet = petRepository.save(pet);

        customer.getPets().add(pet);
        customerRepository.save(customer);

        return pet;
    }

    /**
     * Retrieves a list of pets associated with a specified customer.
     *
     * @param customerId the ID of the customer
     * @return a list of pets associated with the customer
     */
    public List<Pet> getPetsByCustomerId(long customerId) {
        return petRepository.findPetByCustomerId(customerId);
    }

    /**
     * Retrieves a list of all pet entities.
     *
     * @return a list of all pets
     */
    public List<Pet> getAllPets() {
        return petRepository.findAll();
    }

    /**
     * Retrieves a pet entity by its ID.
     *
     * @param petId the ID of the pet
     * @return the retrieved pet entity
     */
    public Pet getPetById(Long petId) {
        return petRepository.getOne(petId);
    }
}