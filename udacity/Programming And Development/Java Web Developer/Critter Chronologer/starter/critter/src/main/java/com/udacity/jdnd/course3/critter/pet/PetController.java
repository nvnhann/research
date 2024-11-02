package com.udacity.jdnd.course3.critter.pet;

import com.udacity.jdnd.course3.critter.entity.Pet;
import com.udacity.jdnd.course3.critter.service.PetService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.server.ResponseStatusException;

import java.util.List;
import java.util.stream.Collectors;

/**
 * Handles web requests related to Pets.
 */
@RestController
@RequestMapping("/pet")
public class PetController {

    @Autowired
    PetService petService;

    /**
     * Converts a Pet entity to a PetDTO.
     *
     * @param pet the Pet entity to convert
     * @return the converted PetDTO
     */
    private PetDTO convertPetToPetDTO(Pet pet) {
        return new PetDTO(pet.getId(), pet.getType(), pet.getName(), pet.getCustomer().getId(), pet.getBirthDate(), pet.getNotes());
    }

    /**
     * Saves a new Pet entity and returns the saved PetDTO.
     *
     * @param petDTO the PetDTO containing the details of the pet to save
     * @return the saved PetDTO
     */
    @PostMapping
    public PetDTO savePet(@RequestBody PetDTO petDTO) {
        Pet pet = new Pet(petDTO.getType(), petDTO.getName(), petDTO.getBirthDate(), petDTO.getNotes());
        try {
            Pet savedPet = petService.savePet(pet, petDTO.getOwnerId());
            return convertPetToPetDTO(savedPet);
        } catch (Exception exception) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "Pet could not be saved", exception);
        }
    }

    /**
     * Retrieves a PetDTO by its ID.
     *
     * @param petId the ID of the pet to retrieve
     * @return the retrieved PetDTO
     */
    @GetMapping("/{petId}")
    public PetDTO getPet(@PathVariable long petId) {
        try {
            Pet pet = petService.getPetById(petId);
            return convertPetToPetDTO(pet);
        } catch (Exception exception) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "Pet with id: " + petId + " not found", exception);
        }
    }

    /**
     * Retrieves a list of all PetDTOs.
     *
     * @return a list of all PetDTOs
     */
    @GetMapping
    public List<PetDTO> getPets() {
        List<Pet> pets = petService.getAllPets();
        return pets.stream().map(this::convertPetToPetDTO).collect(Collectors.toList());
    }

    /**
     * Retrieves a list of PetDTOs by the owner's ID.
     *
     * @param ownerId the ID of the owner
     * @return a list of PetDTOs belonging to the owner
     */
    @GetMapping("/owner/{ownerId}")
    public List<PetDTO> getPetsByOwner(@PathVariable long ownerId) {
        try {
            List<Pet> pets = petService.getPetsByCustomerId(ownerId);
            return pets.stream().map(this::convertPetToPetDTO).collect(Collectors.toList());
        } catch (Exception exception) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "Owner pet with id " + ownerId + " not found", exception);
        }
    }
}