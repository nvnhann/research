package com.udacity.jwdnd.course1.cloudstorage.mapper;

import com.udacity.jwdnd.course1.cloudstorage.model.Note;
import org.apache.ibatis.annotations.*;

import java.util.List;

@Mapper
public interface NoteMapper {

    /**
     * Retrieve all notes from the database.
     *
     * @return a list of all notes
     */
    @Select("SELECT * FROM NOTES")
    List<Note> getAllNotes();

    /**
     * Retrieve a note by its ID.
     *
     * @param noteid the ID of the note to retrieve
     * @return the note with the specified ID
     */
    @Select("SELECT * FROM NOTES WHERE noteid = #{noteid}")
    Note getNoteById(Integer noteid);

    /**
     * Insert a new note into the database.
     *
     * @param note the note to insert
     * @return the number of rows affected
     */
    @Insert("INSERT INTO NOTES(notetitle, notedescription, userid) VALUES (" +
            "#{noteTitle}, #{noteDescription}, #{userId})")
    @Options(useGeneratedKeys = true, keyProperty = "noteId")
    int insert(Note note);

    /**
     * Delete a note by its ID.
     *
     * @param noteid the ID of the note to delete
     */
    @Delete("DELETE FROM NOTES WHERE noteid = #{noteid}")
    void delete(Integer noteid);

    /**
     * Delete all notes from the database.
     */
    @Delete("DELETE FROM NOTES")
    void deleteAll();

    /**
     * Update an existing note in the database.
     *
     * @param noteTitle       the new title of the note
     * @param noteDescription the new description of the note
     * @param noteId          the ID of the note to update
     */
    @Update("UPDATE notes " +
            "SET notetitle = #{noteTitle}, notedescription = #{noteDescription} " +
            "WHERE noteid = #{noteId}")
    void update(@Param("noteTitle") String noteTitle,
               @Param("noteDescription") String noteDescription,
               @Param("noteId") Integer noteId);
}