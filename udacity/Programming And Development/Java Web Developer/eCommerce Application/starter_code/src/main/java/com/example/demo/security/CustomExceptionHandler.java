package com.example.demo.security;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.ControllerAdvice;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.http.HttpStatus;

@ControllerAdvice
@RestController
public class CustomExceptionHandler {

    @ExceptionHandler(JWTAuthenticationVerificationFilter.InvalidTokenException.class)
    public final ResponseEntity<String> handleInvalidTokenException(JWTAuthenticationVerificationFilter.InvalidTokenException ex) {
        return ResponseEntity.status(HttpStatus.UNAUTHORIZED).body(ex.getMessage());
    }
}