package com.udacity.jwdnd.course1.cloudstorage.services;

import com.udacity.jwdnd.course1.cloudstorage.mapper.NoteMapper;
import com.udacity.jwdnd.course1.cloudstorage.mapper.UserMapper;
import com.udacity.jwdnd.course1.cloudstorage.mapper.UserNoteMapper;
import com.udacity.jwdnd.course1.cloudstorage.model.UserNote;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class NoteService {

    private final Logger logger = LoggerFactory.getLogger(NoteService.class);

    private final NoteMapper noteMapper;
    private final UserNoteMapper userNoteMapper;
    private final UserMapper userMapper;

    public NoteService(NoteMapper noteMapper, UserNoteMapper userNoteMapper, UserMapper userMapper) {
        this.noteMapper = noteMapper;
        this.userNoteMapper = userNoteMapper;
        this.userMapper = userMapper;
    }

    /**
     * Retrieves a list of notes for the specified username.
     *
     * @param username the username whose notes are to be retrieved
     * @return a list of UserNote objects
     */
    public List<UserNote> getNotesByUsername(String username) {
        Integer userid = userMapper.getUserIdByUsername(username);
        if (userid == null) {
            return null;
        }
        return this.userNoteMapper.getNotesByUserId(userid);
    }

    /**
     * Inserts or updates a note for the specified user.
     *
     * @param username the username to whom the note belongs
     * @param userNote the note to insert or update
     * @return a boolean indicating success
     */
    public Boolean insertOrUpdateNoteByUser(String username, UserNote userNote) {
        String noteTitle = userNote.getNoteTitle();
        String noteDescription = userNote.getNoteDescription();
        Integer noteId = userNote.getNoteId();

        if (noteId == null || noteId.toString().isEmpty()) {
            Integer userid = userMapper.getUserIdByUsername(username);
            this.userNoteMapper.insertNoteByUserId(userid, noteTitle, noteDescription);
        } else {
            this.noteMapper.update(noteTitle, noteDescription, noteId);
        }
        return true;
    }

    /**
     * Deletes a note with the specified ID.
     *
     * @param noteId   the ID of the note to delete
     * @param username the username to whom the note belongs
     * @return a boolean indicating success
     */
    public Boolean deleteNote(Integer noteId, String username) {
        this.noteMapper.delete(noteId);
        return true;
    }
}