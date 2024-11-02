package com.udacity.jwdnd.course1.cloudstorage.mapper;

import com.udacity.jwdnd.course1.cloudstorage.model.Credential;
import org.apache.ibatis.annotations.*;

@Mapper
public interface CredentialMapper {

    /**
     * Inserts a new credential into the CREDENTIALS table.
     *
     * @param credential the credential to insert
     * @return the number of rows affected
     */
    @Insert("INSERT INTO CREDENTIALS(url, username, key, password, userid) VALUES (" +
            "#{url}, #{username}, #{key}, #{password}, #{userid})")
    @Options(useGeneratedKeys = true, keyProperty = "credentialid")
    int insert(Credential credential);

    /**
     * Retrieves a credential by its credential ID.
     *
     * @param credentialid the ID of the credential to retrieve
     * @return the Credential object
     */
    @Select("SELECT * FROM CREDENTIALS WHERE credentialid = #{credentialid}")
    Credential getCredentialByCredentialId(Integer credentialid);

    /**
     * Deletes a credential by its credential ID.
     *
     * @param credentialid the ID of the credential to delete
     */
    @Delete("DELETE FROM CREDENTIALS WHERE credentialid = #{credentialid}")
    void delete(Integer credentialid);

    /**
     * Updates an existing credential in the CREDENTIALS table.
     *
     * @param url          the new URL
     * @param username     the new username
     * @param key          the new encryption key
     * @param password     the new password
     * @param credentialid the ID of the credential to update
     */
    @Update("UPDATE credentials " +
            "SET url = #{url}, username = #{username}, key = #{key}, " +
            "password = #{password} " +
            "WHERE credentialid = #{credentialid}")
    void update(
            String url, String username, String key, String password, Integer credentialid);
}