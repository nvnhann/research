package com.udacity.jwdnd.course1.cloudstorage.model;

public class Note {

    private Integer noteId;
    private String noteTitle;
    private String noteDescription;
    private Integer userId;

    /**
     * Constructor for creating a Note with noteId, noteTitle, and noteDescription.
     *
     * @param noteId          the ID of the note
     * @param noteTitle       the title of the note
     * @param noteDescription the description of the note
     */
    public Note(Integer noteId, String noteTitle, String noteDescription) {
        this.noteId = noteId;
        this.noteTitle = noteTitle;
        this.noteDescription = noteDescription;
    }

    /**
     * Constructor for creating a Note with noteId, noteTitle, noteDescription, and userId.
     *
     * @param noteId          the ID of the note
     * @param noteTitle       the title of the note
     * @param noteDescription the description of the note
     * @param userId          the ID of the user who owns the note
     */
    public Note(Integer noteId, String noteTitle, String noteDescription, Integer userId) {
        this.noteId = noteId;
        this.noteTitle = noteTitle;
        this.noteDescription = noteDescription;
        this.userId = userId;
    }

    public Integer getNoteId() {
        return noteId;
    }

    public void setNoteId(Integer noteId) {
        this.noteId = noteId;
    }

    public String getNoteTitle() {
        return noteTitle;
    }

    public void setNoteTitle(String noteTitle) {
        this.noteTitle = noteTitle;
    }

    public String getNoteDescription() {
        return noteDescription;
    }

    public void setNoteDescription(String noteDescription) {
        this.noteDescription = noteDescription;
    }

    public Integer getUserId() {
        return userId;
    }

    public void setUserId(Integer userId) {
        this.userId = userId;
    }
}