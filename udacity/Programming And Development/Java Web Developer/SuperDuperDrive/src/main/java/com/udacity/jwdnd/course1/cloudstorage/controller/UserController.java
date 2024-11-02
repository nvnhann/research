package com.udacity.jwdnd.course1.cloudstorage.controller;

import com.udacity.jwdnd.course1.cloudstorage.services.UserService;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestMapping;

/**
 * Controller for handling user-related requests.
 */
@Controller
@RequestMapping("/users")
public class UserController {

    /**
     * Constructor for UserController.
     *
     * @param userService the instance of UserService to manage users
     */
    public UserController(UserService userService) {
    }
}