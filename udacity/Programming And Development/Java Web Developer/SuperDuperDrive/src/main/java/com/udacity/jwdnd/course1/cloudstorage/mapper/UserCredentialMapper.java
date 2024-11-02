package com.udacity.jwdnd.course1.cloudstorage.mapper;

import com.udacity.jwdnd.course1.cloudstorage.model.UserCredential;
import org.apache.ibatis.annotations.Insert;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Options;
import org.apache.ibatis.annotations.Select;

import java.util.List;

@Mapper
public interface UserCredentialMapper {

    /**
     * Retrieves a list of user credentials by username.
     *
     * @param userid the username to search for credentials
     * @return a list of user credentials
     */
    @Select("SELECT * FROM CREDENTIALS WHERE userid = #{userid}")
    List<UserCredential> getCredentialsByUsername(Integer userid);

    /**
     * Inserts a new user credential by username.
     *
     * @param url      the URL associated with the credential
     * @param username the username for the credential
     * @param key      the key used for encryption
     * @param password the encrypted password
     * @param userid   the username to associate the credential with
     */
    @Insert("INSERT INTO CREDENTIALS (url, username, key, password, userid) VALUES (#{url}, #{username}, #{key}, #{password}, #{userid})")
    void insertCredentialByUsername(
            String url,
            String username,
            String key,
            String password,
            Integer userid);
}