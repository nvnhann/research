package com.udacity.jwdnd.course1.cloudstorage.controller;

import com.udacity.jwdnd.course1.cloudstorage.model.UserCredential;
import com.udacity.jwdnd.course1.cloudstorage.model.UserNote;
import com.udacity.jwdnd.course1.cloudstorage.model.UserVO;
import com.udacity.jwdnd.course1.cloudstorage.services.AuthorizationService;
import com.udacity.jwdnd.course1.cloudstorage.services.CredentialService;
import com.udacity.jwdnd.course1.cloudstorage.services.FileService;
import com.udacity.jwdnd.course1.cloudstorage.services.NoteService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.security.core.Authentication;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.servlet.mvc.support.RedirectAttributes;

import java.util.HashMap;
import java.util.Map;

@Controller
public class HomeController {

    private final Logger logger = LoggerFactory.getLogger(HomeController.class);

    private final AuthorizationService authorizationService;
    private final NoteService noteService;
    private final CredentialService credentialService;
    private final FileService fileService;


    public HomeController(
            AuthorizationService authorizationService,
            NoteService noteService,
            CredentialService credentialService,
            FileService fileService
    ) {
        this.authorizationService = authorizationService;
        this.noteService = noteService;
        this.credentialService = credentialService;
        this.fileService = fileService;
    }

    /**
     * Handles the home page request and populates the model with user notes, credentials, and files.
     *
     * @param userNote       the user note model attribute
     * @param userCredential the user credential model attribute
     * @param authentication   the authentication object containing user details
     * @param model            the model to be populated
     * @return the name of the home view template
     */
    @GetMapping("/home")
    public String getHomepage(
            @ModelAttribute("userNote") UserNote userNote,
            @ModelAttribute("userCredential") UserCredential userCredential,
            Authentication authentication,
            Model model
    ) {
        String username = (String) authentication.getPrincipal();
        Map<String, Object> data = new HashMap<>();
        data.put("noteList", this.noteService.getNotesByUsername(username));
        if (this.noteService.getNotesByUsername(username) == null) {
            return "redirect:/login";
        }
        data.put("credentialList", this.credentialService.getCredentialsByUsername(username));
        data.put("fileList", this.fileService.getFilesByUser(username));
        model.addAllAttributes(data);
        return "home";
    }

    /**
     * Handles the logout request and redirects to the login page.
     *
     * @param userVo the user view object
     * @param model  the model to be populated
     * @return the name of the login view template
     */
    @GetMapping("/logout")
    public String logOut(
            @ModelAttribute("userVo") UserVO userVo,
            Model model
    ) {
        this.logger.info("logout");
        return this.loginPage(userVo, false, true, model);
    }

    /**
     * Handles the login page request.
     *
     * @param userVo     the user view object
     * @param errorValue indicates if there was a login error
     * @param loggedOut  indicates if the user has logged out
     * @param model      the model to be populated
     * @return the name of the login view template
     */
    @GetMapping("/login")
    public String loginPage(
            @ModelAttribute("userVo") UserVO userVo,
            @RequestParam(required = false, name = "error") Boolean errorValue,
            @RequestParam(required = false, name = "loggedOut") Boolean loggedOut,
            Model model
    ) {
        Boolean hasError = errorValue != null && errorValue;
        Boolean isLoggedOut = loggedOut != null && loggedOut;
        Boolean signupSuccessfully = (Boolean) model.getAttribute("signupSuccessfully");
        Map<String, Object> data = new HashMap<>();
        data.put("toLogin", true);
        data.put("loginSuccessfully", false);
        data.put("hasError", hasError);
        data.put("isLoggedOut", isLoggedOut);
        data.put("signupSuccessfully", signupSuccessfully != null && signupSuccessfully);
        model.addAllAttributes(data);
        return "login";
    }

    /**
     * Handles the sign-up form request.
     *
     * @param userVo the user view object
     * @param model  the model to be populated
     * @return the name of the signup view template
     */
    @GetMapping("/signup")
    public String signupForm(
            @ModelAttribute("userVo") UserVO userVo,
            Model model
    ) {
        Map<String, Object> data = new HashMap<>();
        data.put("toSignUp", true);
        data.put("signupSuccessfully", false);
        data.put("hasError", false);
        model.addAllAttributes(data);
        return "signup";
    }

    /**
     * Handles the sign-up form submission.
     *
     * @param userVo the user view object
     * @param model  the model to be populated
     * @return the name of the signup view template
     */
    @PostMapping("/signup")
    public String signupSubmit(
            @ModelAttribute("userVo") UserVO userVo,
            RedirectAttributes redirectAttributes,
            Model model
    ) {
        this.logger.info("Received user info from Signup Form: {}", userVo.toString());
        Map<String, Object> data = new HashMap<>();
        if (!this.authorizationService.signupUser(userVo)) {
            data.put("toSignUp", true);
            data.put("signupSuccessfully", false);
            data.put("hasError", true);
        } else {
            data.put("toSignUp", false);
            data.put("signupSuccessfully", true);
            data.put("hasError", false);
        }
        redirectAttributes.addFlashAttribute("signupSuccessfully", data.get("signupSuccessfully"));
        model.mergeAttributes(data);
        return "redirect:/login";
    }

    /**
     * Handles the result page request.
     *
     * @param authentication the authentication object containing user details
     * @param isSuccess      indicates if the operation was successful
     * @param errorType      the error type if the operation failed
     * @param model          the model to be populated
     * @return the name of the result view template
     */
    @GetMapping("/result")
    public String showResult(
            Authentication authentication,
            @RequestParam(required = false, name = "isSuccess") Boolean isSuccess,
            @RequestParam(required = false, name = "errorType") Integer errorType,
            Model model
    ) {
        Map<String, Object> data = new HashMap<>();
        data.put("isSuccess", isSuccess);
        data.put("errorType", errorType);
        model.addAllAttributes(data);
        return "result";
    }
}