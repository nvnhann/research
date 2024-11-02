package com.udacity.jwdnd.course1.cloudstorage.model;

public class UserNote {

    private Integer userId;
    private Integer noteId;
    private String noteTitle;
    private String noteDescription;

    public UserNote() {
    }

    /**
     * Constructor for UserNote.
     *
     * @param userId          the ID of the user
     * @param noteId          the ID of the note
     * @param noteTitle       the title of the note
     * @param noteDescription the description of the note
     */
    public UserNote(Integer userId, Integer noteId, String noteTitle, String noteDescription) {
        this.userId = userId;
        this.noteId = noteId;
        this.noteTitle = noteTitle;
        this.noteDescription = noteDescription;
    }

    /**
     * Gets the user ID.
     *
     * @return the user ID
     */
    public Integer getUserId() {
        return userId;
    }

    /**
     * Sets the user ID.
     *
     * @param userId the new user ID
     */
    public void setUserId(Integer userId) {
        this.userId = userId;
    }

    /**
     * Gets the note ID.
     *
     * @return the note ID
     */
    public Integer getNoteId() {
        return noteId;
    }

    /**
     * Sets the note ID.
     *
     * @param noteId the new note ID
     */
    public void setNoteId(Integer noteId) {
        this.noteId = noteId;
    }

    /**
     * Gets the note title.
     *
     * @return the note title
     */
    public String getNoteTitle() {
        return noteTitle;
    }

    /**
     * Sets the note title.
     *
     * @param noteTitle the new note title
     */
    public void setNoteTitle(String noteTitle) {
        this.noteTitle = noteTitle;
    }

    /**
     * Gets the note description.
     *
     * @return the note description
     */
    public String getNoteDescription() {
        return noteDescription;
    }

    /**
     * Sets the note description.
     *
     * @param noteDescription the new note description
     */
    public void setNoteDescription(String noteDescription) {
        this.noteDescription = noteDescription;
    }

    @Override
    public String toString() {
        return "UserNote{" +
                "userId=" + userId +
                ", noteId=" + noteId +
                ", noteTitle='" + noteTitle + '\'' +
                ", noteDescription='" + noteDescription + '\'' +
                '}';
    }
}