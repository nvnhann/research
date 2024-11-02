package com.udacity.jwdnd.course1.cloudstorage.mapper;

import com.udacity.jwdnd.course1.cloudstorage.model.User;
import org.apache.ibatis.annotations.*;

import java.util.List;

@Mapper
public interface UserMapper {

    /**
     * Retrieves all users.
     *
     * @return a list of users
     */
    @Select("SELECT * FROM USERS")
    List<User> getAllUsers();

    /**
     * Retrieves a user by username.
     *
     * @param username the username
     * @return the user with the specified username
     */
    @Select("SELECT * FROM USERS WHERE username = #{username}")
    User getUserByUsername(String username);

    /**
     * Retrieves a user by user ID.
     *
     * @param userid the user ID
     * @return the user with the specified user ID
     */
    @Select("SELECT * FROM USERS WHERE userid = #{userid}")
    User getUserById(Integer userid);

    /**
     * Inserts a new user.
     *
     * @param user the user to insert
     * @return the number of rows affected
     */
    @Insert("INSERT INTO USERS(username,salt,password,firstname,lastname) VALUES (" +
            "#{username}, #{salt}, #{password}, #{firstname}, #{lastname})")
    @Options(useGeneratedKeys = true, keyProperty = "userid")
    int insert(User user);

    /**
     * Deletes a user by user ID.
     *
     * @param userid the user ID
     */
    @Delete("DELETE FROM USERS WHERE userid = #{userid}")
    void delete(Integer userid);

    /**
     * Retrieves the user ID for a given username.
     *
     * @param username the username
     * @return the user ID associated with the specified username
     */
    @Select("SELECT userid FROM USERS WHERE username = #{username}")
    Integer getUserIdByUsername(@Param("username") String username);
}