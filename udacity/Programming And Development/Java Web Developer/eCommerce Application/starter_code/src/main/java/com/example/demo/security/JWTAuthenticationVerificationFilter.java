package com.example.demo.security;

import com.auth0.jwt.JWT;
import com.auth0.jwt.algorithms.Algorithm;
import com.auth0.jwt.exceptions.JWTVerificationException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpStatus;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.web.authentication.www.BasicAuthenticationFilter;
import org.springframework.stereotype.Component;
import org.springframework.web.bind.annotation.ResponseStatus;

import javax.servlet.FilterChain;
import javax.servlet.ServletException;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;
import java.util.ArrayList;

import static com.example.demo.config.Constant.*;


@Component
public class JWTAuthenticationVerificationFilter extends BasicAuthenticationFilter {

    private static final Logger logger = LoggerFactory.getLogger(JWTAuthenticationVerificationFilter.class);

    public JWTAuthenticationVerificationFilter(AuthenticationManager authManager) {
        super(authManager);
    }

    @Override
    protected void doFilterInternal(HttpServletRequest req,
                                    HttpServletResponse res,
                                    FilterChain chain) throws IOException, ServletException {
        String header = req.getHeader(HEADER_STRING);
        logger.debug("Header: {}", header);

        if (header == null || !header.startsWith(TOKEN_PREFIX)) {
            logger.warn("Invalid header or token prefix.");
            chain.doFilter(req, res);
            return;
        }
        try {
            UsernamePasswordAuthenticationToken authentication = getAuthentication(req);

            SecurityContextHolder.getContext().setAuthentication(authentication);

        } catch (InvalidTokenException e) {
            SecurityContextHolder.clearContext();
            res.sendError(HttpServletResponse.SC_UNAUTHORIZED, e.getMessage());
            return;
        }
        chain.doFilter(req, res);
    }

    private UsernamePasswordAuthenticationToken getAuthentication(HttpServletRequest request) {
        String token = request.getHeader(HEADER_STRING);
        logger.debug("Token: {}", token);
        if (token != null) {
            try {
                // Parse the token.
                String user = JWT.require(Algorithm.HMAC512(SECRET.getBytes()))
                        .build()
                        .verify(token.replace(TOKEN_PREFIX, ""))
                        .getSubject();
                logger.debug("User: {}", user);

                if (user != null) {
                    return new UsernamePasswordAuthenticationToken(user, null, new ArrayList<>());
                }
            } catch (JWTVerificationException ex) {
                logger.error("Invalid token: {}", ex.getMessage());
                throw new InvalidTokenException("Invalid token");
            }
        }
        return null;
    }
    @ResponseStatus(HttpStatus.UNAUTHORIZED)
    public class InvalidTokenException extends RuntimeException {
        public InvalidTokenException(String message) {
            super(message);
        }
    }
}
