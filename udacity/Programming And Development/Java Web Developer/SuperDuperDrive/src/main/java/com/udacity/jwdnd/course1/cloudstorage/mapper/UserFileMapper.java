package com.udacity.jwdnd.course1.cloudstorage.mapper;

import com.udacity.jwdnd.course1.cloudstorage.model.UserFile;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Select;

import java.util.List;
import java.util.Map;

@Mapper
public interface UserFileMapper {

    /**
     * Retrieves a list of UserFiles by the username.
     *
     * @param userid the username for which to retrieve files
     * @return a list of UserFiles associated with the given username
     */
    @Select("SELECT * FROM FILES WHERE userid = #{userid}")
    List<UserFile> getFileByUserId(Integer userid);

    /**
     * Retrieves a list of UserFiles by the username and file name.
     *
     * @param fileName a map containing the parameters "username" and "fileName"
     * @return a list of UserFiles associated with the given username and file name
     */
    @Select("SELECT * FROM FILES WHERE userid = #{userid} AND filename = #{fileName}")
    List<UserFile> getFileByUsernameAndFileName(String fileName, Integer userid);
}