package com.example.demo.controller;

import com.example.demo.controllers.UserController;
import com.example.demo.model.persistence.Cart;
import com.example.demo.model.persistence.User;
import com.example.demo.model.persistence.repositories.CartRepository;
import com.example.demo.model.persistence.repositories.UserRepository;
import com.example.demo.model.requests.CreateUserRequest;
import org.junit.Before;
import org.junit.Test;
import org.springframework.http.ResponseEntity;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;

import java.lang.reflect.Field;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class UserControllerTest {

    private UserController userController;

    private final UserRepository userRepository = mock(UserRepository.class);

    private final CartRepository cartRepository = mock(CartRepository.class);

    private final BCryptPasswordEncoder bCryptPasswordEncoder = mock(BCryptPasswordEncoder.class);

    @Before
    public void setUp() {
        userController = new UserController();
        injectDependencies(userController, "userRepository", userRepository);
        injectDependencies(userController, "cartRepository", cartRepository);
        injectDependencies(userController, "bCryptPasswordEncoder", bCryptPasswordEncoder);

    }

    /**
     * Injects dependencies into the target object.
     *
     * @param target     The object into which the dependency should be injected.
     * @param fieldName  The name of the field to inject.
     * @param dependency The dependency to be injected.
     */
    private static void injectDependencies(Object target, String fieldName, Object dependency) {
        try {
            Field field = target.getClass().getDeclaredField(fieldName);
            boolean accessible = field.isAccessible();
            field.setAccessible(true);
            field.set(target, dependency);
            field.setAccessible(accessible);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            throw new RuntimeException(e);
        }
    }

    @Test
    public void createUser() {
        when(bCryptPasswordEncoder.encode("password")).thenReturn("hashed");
        CreateUserRequest createUserRequest = new CreateUserRequest();
        createUserRequest.setUsername("username");
        createUserRequest.setPassword("password");
        createUserRequest.setConfirmPassword("password");

        final ResponseEntity<User> response = userController.createUser(createUserRequest);

        assertNotNull(response);
        assertEquals(200,response.getStatusCodeValue());

        User user = response.getBody();
        assertNotNull(user);
        assertEquals(0, user.getId());
        assertEquals("username", user.getUsername());
        assertEquals("hashed", user.getPassword());
    }

    @Test
    public void createUserInvalidPassword() throws Exception {
        when(bCryptPasswordEncoder.encode("password")).thenReturn("hashed");
        CreateUserRequest createUserRequest = new CreateUserRequest();
        createUserRequest.setUsername("username");
        createUserRequest.setPassword("password");
        createUserRequest.setConfirmPassword("password1");
        final ResponseEntity<User> response = userController.createUser(createUserRequest);
        assertNotNull(response);
        assertEquals(400,response.getStatusCodeValue());
    }

    @Test
    public void getUsernameById() {
        User user = new User();
        user.setUsername("username");
        Cart cart = new Cart();
        cart.setId((long) 0);
        cart.setUser(user);
        user.setCart(cart);
        user.setId(0);
        user.setPassword("password");

        when(userRepository.findById((long) 0)).thenReturn(java.util.Optional.of(user));
        final ResponseEntity<User> response = userController.findById((long) 0);

        assertNotNull(response);
        assertEquals(200,response.getStatusCodeValue());

        User actualUser = response.getBody();
        assertNotNull(actualUser);
        assertEquals(0, actualUser.getId());
        assertEquals("username", actualUser.getUsername());
        assertEquals("password", actualUser.getPassword());
        assertEquals(cart, actualUser.getCart());
    }

    @Test
    public void getUserByUsername() {
        User user = new User();
        user.setUsername("username");
        Cart cart = new Cart();
        cart.setId((long) 0);
        cart.setUser(user);
        user.setCart(cart);
        user.setId(0);
        user.setPassword("password");

        when(userRepository.findByUsername("username")).thenReturn(user);
        final ResponseEntity<User> response = userController.findByUserName("username");
        assertNotNull(response);
        assertEquals(200,response.getStatusCodeValue());

        User actualUser = response.getBody();
        assertNotNull(actualUser);
        assertEquals(0, actualUser.getId());
        assertEquals("username", actualUser.getUsername());
        assertEquals("password", actualUser.getPassword());
        assertEquals(cart, actualUser.getCart());
    }

}