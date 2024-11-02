package com.udacity.jwdnd.course1.cloudstorage.model;

/**
 * Represents a file entity with various attributes and methods to manipulate them.
 */
public class File {

    private Integer fileid;
    private String filename;
    private String contenttype;
    private String filesize;
    private Integer userid;
    private byte[] filedata;

    /**
     * Constructs a new File object with specified attributes.
     *
     * @param fileid      the file ID
     * @param filename    the name of the file
     * @param contenttype the content type of the file
     * @param filesize    the size of the file
     * @param filedata    the data of the file
     */
    public File(
            Integer fileid,
            String filename,
            String contenttype,
            String filesize,
            byte[] filedata) {

        this(fileid, filename, contenttype, filesize, null, filedata);
    }

    /**
     * Constructs a new File object with specified attributes.
     *
     * @param fileid      the file ID
     * @param filename    the name of the file
     * @param contenttype the content type of the file
     * @param filesize    the size of the file
     * @param userid      the user ID associated with the file
     * @param filedata    the data of the file
     */
    public File(
            Integer fileid,
            String filename,
            String contenttype,
            String filesize,
            Integer userid,
            byte[] filedata) {

        this.fileid = fileid;
        this.filename = filename;
        this.contenttype = contenttype;
        this.filesize = filesize;
        this.userid = userid;
        this.filedata = filedata;
    }

    public Integer getFileid() {
        return fileid;
    }

    public void setFileid(Integer fileid) {
        this.fileid = fileid;
    }

    public String getFilename() {
        return filename;
    }

    public void setFilename(String filename) {
        this.filename = filename;
    }

    public String getContenttype() {
        return contenttype;
    }

    public void setContenttype(String contenttype) {
        this.contenttype = contenttype;
    }

    public String getFilesize() {
        return filesize;
    }

    public void setFilesize(String filesize) {
        this.filesize = filesize;
    }

    public Integer getUserid() {
        return userid;
    }

    public void setUserid(Integer userid) {
        this.userid = userid;
    }

    public byte[] getFiledata() {
        return filedata;
    }

    public void setFiledata(byte[] filedata) {
        this.filedata = filedata;
    }
}