package com.udacity.jwdnd.course1.cloudstorage.model;

public class UserFile {

    private Integer fileId;
    private String fileName;
    private String contentType;
    private String fileSize;
    private Integer userId;
    private byte[] fileData;

    /**
     * Constructor for UserFile.
     *
     * @param fileId      the file ID
     * @param fileName    the name of the file
     * @param contentType the type of content of the file
     * @param fileSize    the size of the file
     * @param userId      the ID of the user
     * @param fileData    the data of the file
     */
    public UserFile(
            Integer fileId,
            String fileName,
            String contentType,
            String fileSize,
            Integer userId,
            byte[] fileData) {

        this.fileId = fileId;
        this.fileName = fileName;
        this.contentType = contentType;
        this.fileSize = fileSize;
        this.userId = userId;
        this.fileData = fileData;
    }

    /**
     * Gets the file ID.
     *
     * @return the file ID
     */
    public Integer getFileId() {
        return fileId;
    }

    /**
     * Sets the file ID.
     *
     * @param fileId the file ID to set
     */
    public void setFileId(Integer fileId) {
        this.fileId = fileId;
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
     * @param userId the user ID to set
     */
    public void setUserId(Integer userId) {
        this.userId = userId;
    }

    /**
     * Gets the file name.
     *
     * @return the file name
     */
    public String getFileName() {
        return fileName;
    }

    /**
     * Sets the file name.
     *
     * @param fileName the file name to set
     */
    public void setFileName(String fileName) {
        this.fileName = fileName;
    }

    /**
     * Gets the content type.
     *
     * @return the content type
     */
    public String getContentType() {
        return contentType;
    }

    /**
     * Sets the content type.
     *
     * @param contentType the content type to set
     */
    public void setContentType(String contentType) {
        this.contentType = contentType;
    }

    /**
     * Gets the file size.
     *
     * @return the file size
     */
    public String getFileSize() {
        return fileSize;
    }

    /**
     * Sets the file size.
     *
     * @param fileSize the file size to set
     */
    public void setFileSize(String fileSize) {
        this.fileSize = fileSize;
    }

    /**
     * Gets the file data.
     *
     * @return the file data
     */
    public byte[] getFileData() {
        return fileData;
    }

    /**
     * Sets the file data.
     *
     * @param fileData the file data to set
     */
    public void setFileData(byte[] fileData) {
        this.fileData = fileData;
    }
}