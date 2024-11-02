package com.udacity.jwdnd.course1.cloudstorage.services;

import com.udacity.jwdnd.course1.cloudstorage.mapper.UserMapper;
import com.udacity.jwdnd.course1.cloudstorage.model.User;
import com.udacity.jwdnd.course1.cloudstorage.model.UserVO;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.security.SecureRandom;
import java.util.Base64;

@Service
public class UserService {

    private final Logger logger = LoggerFactory.getLogger(UserService.class);
    private final UserMapper userMapper;
    private final HashService hashService;

    public UserService(UserMapper userMapper, HashService hashService) {
        this.userMapper = userMapper;
        this.hashService = hashService;
    }

    /**
     * Checks if a username is available.
     *
     * @param username the username to check
     * @return true if username is available, false otherwise
     */
    public boolean isUsernameAvailable(String username) {
        return this.userMapper.getUserByUsername(username) == null;
    }

    /**
     * Creates a new user with the given details.
     *
     * @param userVo the UserVO object containing user details
     * @return the number of rows affected
     */
    public int createUser(UserVO userVo) {
        byte[] salt = generateSalt();
        String encodedSalt = Base64.getEncoder().encodeToString(salt);
        String hashedPassword = hashService.getHashedValue(userVo.getPassword(), encodedSalt);

        User newUser = new User(
                null,
                userVo.getUsername(),
                encodedSalt,
                hashedPassword,
                userVo.getFirstName(),
                userVo.getLastName());
        return userMapper.insert(newUser);
    }

    /**
     * Retrieves a user by username.
     *
     * @param username the username to search for
     * @return the User object if found, otherwise null
     */
    public User getUser(String username) {
        return this.userMapper.getUserByUsername(username);
    }

    /**
     * Generates a random salt.
     *
     * @return a byte array representing the salt
     */
    private byte[] generateSalt() {
        SecureRandom random = new SecureRandom();
        byte[] salt = new byte[16];
        random.nextBytes(salt);
        return salt;
    }
}