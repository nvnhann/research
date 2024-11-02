package com.udacity.jwdnd.course1.cloudstorage.controller;

import com.udacity.jwdnd.course1.cloudstorage.model.File;
import com.udacity.jwdnd.course1.cloudstorage.services.FileService;
import org.springframework.core.io.InputStreamResource;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.Authentication;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.multipart.MultipartFile;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;

@Controller
@RequestMapping("/files")
public class FileController {

    private final FileService fileService;

    public FileController(FileService fileService) {
        this.fileService = fileService;
    }

    /**
     * Handles file upload requests.
     *
     * @param fileUpload     the file to upload
     * @param authentication the authentication object containing user details
     * @return the redirection URL indicating the success or failure of the upload
     */
    @PostMapping("/upload")
    public String uploadFile(
            @RequestParam("fileUpload") MultipartFile fileUpload,
            Authentication authentication
    ) {
        String username = (String) authentication.getPrincipal();

        if (fileUpload.isEmpty()) {
            return "redirect:/result?isSuccess=" + false + "&errorType=" + 1;
        }

        String fileName = fileUpload.getOriginalFilename();

        if (!this.fileService.isFileNameAvailableForUser(username, fileName)) {
            return "redirect:/result?isSuccess=" + false + "&errorType=" + 2;
        }

        try {
            this.fileService.saveFile(fileUpload, username);
        } catch (IOException e) {
            e.printStackTrace();
            return "redirect:/result?isSuccess=" + false;
        }

        return "redirect:/result?isSuccess=" + true;
    }

    /**
     * Handles file deletion requests.
     *
     * @param fileId the ID of the file to delete
     * @return the redirection URL indicating the success or failure of the deletion
     */
    @GetMapping("/delete")
    public String deleteFile(@RequestParam(required = false, name = "fileId") Integer fileId) {
        Boolean isSuccess = this.fileService.deleteFile(fileId);
        return "redirect:/result?isSuccess=" + isSuccess;
    }

    /**
     * Handles file download requests.
     *
     * @param fileId the ID of the file to download
     * @return the ResponseEntity containing the file data
     */
    @GetMapping("/download")
    public ResponseEntity<InputStreamResource> downloadFile(@RequestParam(required = false, name = "fileId") Integer fileId) {
        File file = this.fileService.getFileByFileId(fileId);

        InputStreamResource resource = new InputStreamResource(new ByteArrayInputStream(file.getFiledata()));

        return ResponseEntity.ok()
                .header(HttpHeaders.CONTENT_DISPOSITION, "attachment;filename=" + file.getFilename())
                .contentType(MediaType.parseMediaType(file.getContenttype()))
                .body(resource);
    }
}