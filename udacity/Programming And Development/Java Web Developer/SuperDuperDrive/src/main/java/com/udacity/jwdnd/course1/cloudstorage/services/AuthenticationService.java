package com.udacity.jwdnd.course1.cloudstorage.services;

import com.udacity.jwdnd.course1.cloudstorage.mapper.CredentialMapper;
import com.udacity.jwdnd.course1.cloudstorage.mapper.UserMapper;
import com.udacity.jwdnd.course1.cloudstorage.model.User;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.security.authentication.AuthenticationProvider;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.AuthenticationException;
import org.springframework.stereotype.Service;

import java.util.ArrayList;

@Service
public class AuthenticationService implements AuthenticationProvider {

    private final Logger logger = LoggerFactory.getLogger(AuthenticationService.class);

    private final UserMapper userMapper;
    private final HashService hashService;

    /**
     * Constructor for AuthenticationService.
     *
     * @param credentialMapper the CredentialMapper for interacting with credentials
     * @param userMapper       the UserMapper for interacting with users
     * @param hashService      the HashService for hashing passwords
     */
    public AuthenticationService(
            CredentialMapper credentialMapper,
            UserMapper userMapper,
            HashService hashService) {
        this.userMapper = userMapper;
        this.hashService = hashService;
    }

    /**
     * Authenticates the user based on the provided credentials.
     *
     * @param authentication the authentication request object
     * @return the authenticated user token if authentication is successful, null otherwise
     * @throws AuthenticationException if an authentication error occurs
     */
    @Override
    public Authentication authenticate(
            Authentication authentication) throws AuthenticationException {

        String username = authentication.getName();
        String password = authentication.getCredentials().toString();

        User user = this.userMapper.getUserByUsername(username);

        if (user != null) {

            String encodedSalt = user.getSalt();
            String hashedPassword = hashService.getHashedValue(password, encodedSalt);

            if (user.getPassword().equals(hashedPassword)) {
                return new UsernamePasswordAuthenticationToken(
                        username, password, new ArrayList<>());
            }
        }

        return null;
    }

    /**
     * Checks if this provider supports the given authentication type.
     *
     * @param authentication the class of the authentication type
     * @return true if the authentication type is supported, false otherwise
     */
    @Override
    public boolean supports(Class<?> authentication) {
        return authentication.equals(UsernamePasswordAuthenticationToken.class);
    }
}