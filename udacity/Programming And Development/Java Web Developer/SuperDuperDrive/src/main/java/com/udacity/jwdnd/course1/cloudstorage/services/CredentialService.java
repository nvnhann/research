package com.udacity.jwdnd.course1.cloudstorage.services;

import com.udacity.jwdnd.course1.cloudstorage.mapper.CredentialMapper;
import com.udacity.jwdnd.course1.cloudstorage.mapper.UserCredentialMapper;
import com.udacity.jwdnd.course1.cloudstorage.mapper.UserMapper;
import com.udacity.jwdnd.course1.cloudstorage.model.UserCredential;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.security.SecureRandom;
import java.util.Base64;
import java.util.List;
import java.util.stream.Collectors;

@Service
public class CredentialService {

    private final Logger logger = LoggerFactory.getLogger(CredentialService.class);

    private final EncryptionService encryptionService;
    private final CredentialMapper credentialMapper;
    private final UserCredentialMapper userCredentialMapper;
    private final UserMapper userMapper;

    public CredentialService(
            EncryptionService encryptionService,
            CredentialMapper credentialMapper,
            UserCredentialMapper userCredentialMapper,
            UserMapper userMapper
    ) {
        this.encryptionService = encryptionService;
        this.credentialMapper = credentialMapper;
        this.userCredentialMapper = userCredentialMapper;
        this.userMapper = userMapper;
    }

    /**
     * Retrieves the credentials associated with the given username, decrypting the passwords.
     *
     * @param username the username for which to retrieve credentials
     * @return a list of UserCredential objects with decrypted passwords
     */
    public List<UserCredential> getCredentialsByUsername(String username) {
        Integer userid = userMapper.getUserIdByUsername(username);
        List<UserCredential> userCredentialList = userCredentialMapper.getCredentialsByUsername(userid);
        return userCredentialList.stream().peek(this::decryptPassword).collect(Collectors.toList());
    }

    /**
     * Inserts or updates the given UserCredential.
     *
     * @param userCredential the UserCredential object to insert or update
     * @param username       the username associated with the UserCredential
     * @return true if the operation was successful, false otherwise
     */
    public Boolean insertOrUpdateCredential(UserCredential userCredential, String username) {
        Integer credentialId = userCredential.getCredentialId();
        String encryptedPassword = encryptPassword(userCredential);
        userCredential.setPassword(encryptedPassword);

        if (credentialId == null) {
            userCredentialMapper.insertCredentialByUsername(
                    userCredential.getUrl(),
                    userCredential.getUsername(),
                    userCredential.getKey(),
                    userCredential.getPassword(),
                    userMapper.getUserIdByUsername(username));
        } else {
            credentialMapper.update(
                    userCredential.getUrl(),
                    userCredential.getUsername(),
                    userCredential.getKey(),
                    userCredential.getPassword(),
                    credentialId);
        }
        return true;
    }

    /**
     * Deletes the credential associated with the given credential ID.
     *
     * @param credentialId the ID of the credential to delete
     * @return true if the operation was successful, false otherwise
     */
    public Boolean deleteCredential(Integer credentialId) {
        credentialMapper.delete(credentialId);
        return true;
    }

    /**
     * Encrypts the password of the given UserCredential object and updates its key.
     *
     * @param userCredential the UserCredential object containing the password to encrypt
     * @return the encrypted password as a string
     */
    private String encryptPassword(UserCredential userCredential) {
        SecureRandom random = new SecureRandom();
        byte[] key = new byte[16];
        random.nextBytes(key);
        String encodedKey = Base64.getEncoder().encodeToString(key);
        userCredential.setKey(encodedKey);
        return encryptionService.encryptValue(userCredential.getPassword(), encodedKey);
    }

    /**
     * Decrypts the password of the given UserCredential object using its key.
     *
     * @param userCredential the UserCredential object containing the encrypted password
     */
    private void decryptPassword(UserCredential userCredential) {
        String decodedPassword = encryptionService.decryptValue(userCredential.getPassword(), userCredential.getKey());
        userCredential.setDecryptedPassword(decodedPassword);
    }
}