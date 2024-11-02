package com.udacity.jwdnd.course1.cloudstorage.controller;

import com.udacity.jwdnd.course1.cloudstorage.model.UserCredential;
import com.udacity.jwdnd.course1.cloudstorage.services.CredentialService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.security.core.Authentication;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;

/**
 * Controller class for managing user credentials.
 */
@Controller
@RequestMapping("/credentials")
public class CredentialController {

    private final Logger logger = LoggerFactory.getLogger(CredentialController.class);
    private final CredentialService credentialService;

    public CredentialController(CredentialService credentialService) {
        this.credentialService = credentialService;
    }

    /**
     * Handles the submission of a user credential, inserting or updating it.
     *
     * @param userCredential the UserCredential object to insert or update
     * @param authentication   the authentication object containing user details
     * @param model            the model object for the view
     * @return a redirect URL based on the success of the operation
     */
    @PostMapping("/credential")
    public String submitCredential(
            @ModelAttribute("userCredential") UserCredential userCredential,
            Authentication authentication,
            Model model
    ) {
        String username = (String) authentication.getPrincipal();
        Boolean isSuccess = credentialService.insertOrUpdateCredential(userCredential, username);
        return "redirect:/result?isSuccess=" + isSuccess + "#nav-credentials";
    }

    /**
     * Handles the deletion of a user credential.
     *
     * @param userCredentialVO the UserCredential object to delete (not actually used)
     * @param credentialId     the ID of the credential to delete
     * @param authentication   the authentication object containing user details
     * @param model            the model object for the view
     * @return a redirect URL based on the success of the operation
     */
    @GetMapping("/credential")
    public String deleteCredential(
            @ModelAttribute("userCredential") UserCredential userCredentialVO,
            @RequestParam(required = false, name = "credentialId") Integer credentialId,
            Authentication authentication,
            Model model
    ) {
        logger.info("CredentialId: {}", credentialId);
        Boolean isSuccess = credentialService.deleteCredential(credentialId);
        return "redirect:/result?isSuccess=" + isSuccess + "#nav-credentials";
    }
}