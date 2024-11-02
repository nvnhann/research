package com.udacity.jwdnd.course1.cloudstorage.services;

import com.udacity.jwdnd.course1.cloudstorage.mapper.FileMapper;
import com.udacity.jwdnd.course1.cloudstorage.mapper.UserFileMapper;
import com.udacity.jwdnd.course1.cloudstorage.mapper.UserMapper;
import com.udacity.jwdnd.course1.cloudstorage.model.File;
import com.udacity.jwdnd.course1.cloudstorage.model.User;
import com.udacity.jwdnd.course1.cloudstorage.model.UserFile;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Service
public class FileService {

    private final Logger logger = LoggerFactory.getLogger(FileService.class);

    private final FileMapper fileMapper;
    private final UserFileMapper userFileMapper;
    private final UserMapper userMapper;

    public FileService(FileMapper fileMapper, UserFileMapper userFileMapper, UserMapper userMapper) {
        this.fileMapper = fileMapper;
        this.userFileMapper = userFileMapper;
        this.userMapper = userMapper;
    }

    /**
     * Checks if a filename is available for a specific user.
     *
     * @param username the username of the user
     * @param filename the filename to check
     * @return true if the filename is available, false otherwise
     */
    public Boolean isFileNameAvailableForUser(String username, String filename) {
        Map<String, Object> paraMap = new HashMap<>();
        Integer userid = userMapper.getUserIdByUsername(username);
        paraMap.put("userid", userid);
        paraMap.put("filename", filename);
        return this.userFileMapper.getFileByUsernameAndFileName(filename, userid).isEmpty();
    }

    /**
     * Retrieves the list of files for a specific user.
     *
     * @param username the username of the user
     * @return the list of UserFile objects
     */
    public List<UserFile> getFilesByUser(String username) {
        Integer userid = userMapper.getUserIdByUsername(username);
        return this.userFileMapper.getFileByUserId(userid);
    }

    /**
     * Saves a file for a specific user.
     *
     * @param file     the MultipartFile to save
     * @param username the username of the user
     * @return true if the file is saved successfully, false otherwise
     * @throws IOException if an I/O error occurs
     */
    public Boolean saveFile(MultipartFile file, String username) throws IOException {
        User user = this.userMapper.getUserByUsername(username);
        byte[] fileData = file.getBytes();
        String contentType = file.getContentType();
        String fileSize = String.valueOf(file.getSize());
        String fileName = file.getOriginalFilename();
        this.fileMapper.insert(new File(null, fileName, contentType, fileSize, user.getUserid(), fileData));
        return true;
    }

    /**
     * Deletes a file by its file ID.
     *
     * @param fileId the ID of the file to delete
     * @return true if the file is deleted successfully, false otherwise
     */
    public Boolean deleteFile(Integer fileId) {
        this.fileMapper.delete(fileId);
        return true;
    }

    /**
     * Retrieves a file by its file ID.
     *
     * @param fileId the ID of the file to retrieve
     * @return the File object
     */
    public File getFileByFileId(Integer fileId) {
        return this.fileMapper.getFileById(fileId);
    }
}