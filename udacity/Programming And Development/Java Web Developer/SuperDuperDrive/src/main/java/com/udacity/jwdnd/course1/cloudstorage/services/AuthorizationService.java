package com.udacity.jwdnd.course1.cloudstorage.services;

import com.udacity.jwdnd.course1.cloudstorage.model.User;
import com.udacity.jwdnd.course1.cloudstorage.model.UserVO;
import org.springframework.stereotype.Service;

@Service
public class AuthorizationService {

    private final UserService userService;

    public AuthorizationService(UserService userService) {
        this.userService = userService;
    }

    /**
     * Signs up a new user.
     *
     * @param userVO the UserVO object containing user details.
     * @return true if the user was successfully signed up, false otherwise.
     */
    public boolean signupUser(UserVO userVO) {
        String username = userVO.getUsername();

        if (!userService.isUsernameAvailable(username)) {
            return false;
        }

        userService.createUser(userVO);

        return true;
    }
}