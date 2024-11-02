package com.udacity.jwdnd.course1.cloudstorage.mapper;

import com.udacity.jwdnd.course1.cloudstorage.model.File;
import org.apache.ibatis.annotations.*;

import java.util.List;

@Mapper
public interface FileMapper {

    /**
     * Inserts a new file record into the FILES table.
     *
     * @param file the file object to be inserted
     * @return the number of rows affected
     */
    @Insert("INSERT INTO FILES(" +
            "filename, " +
            "contenttype, " +
            "filesize, " +
            "filedata, " +
            "userid) VALUES (" +
            "#{filename}, " +
            "#{contenttype}, " +
            "#{filesize}, " +
            "#{filedata}, " +
            "#{userid})")
    @Options(useGeneratedKeys = true, keyProperty = "fileid")
    int insert(File file);

    /**
     * Retrieves a file record by its file ID.
     *
     * @param fileid the ID of the file
     * @return the file object
     */
    @Select("SELECT * FROM FILES WHERE fileid = #{fileid}")
    File getFileById(Integer fileid);

    /**
     * Retrieves all file records.
     *
     * @return a list of all file objects
     */
    @Select("SELECT * FROM FILES")
    List<File> getAllFiles();

    /**
     * Deletes a file record by its file ID.
     *
     * @param fileid the ID of the file to delete
     */
    @Delete("DELETE FROM FILES WHERE fileid = #{fileid}")
    void delete(Integer fileid);
}