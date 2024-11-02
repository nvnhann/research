package com.udacity.jwdnd.course1.cloudstorage.mapper;

import com.udacity.jwdnd.course1.cloudstorage.model.UserNote;
import org.apache.ibatis.annotations.Insert;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Options;
import org.apache.ibatis.annotations.Select;

import java.util.List;

@Mapper
public interface UserNoteMapper {

    /**
     * Retrieves a list of notes for the given username.
     *
     * @param userid the userid whose notes are to be retrieved
     * @return a list of UserNote objects
     */
    @Select("SELECT * FROM NOTES WHERE userid = #{userid}")
    List<UserNote> getNotesByUserId(Integer userid);

    /**
     * Inserts a new note for the given username.
     *
     * @param userid        the username to whom the note belongs
     * @param notetitle       the title of the note
     * @param notedescription the description of the note
     */
    @Insert("INSERT INTO NOTES (userid, notetitle, notedescription) VALUES (#{userid}, #{notetitle}, #{notedescription})")
    void insertNoteByUserId(
            Integer userid,
            String notetitle,
            String notedescription);
}