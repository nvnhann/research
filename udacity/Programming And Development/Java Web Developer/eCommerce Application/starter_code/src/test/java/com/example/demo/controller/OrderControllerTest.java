package com.example.demo.controller;

import com.example.demo.controllers.OrderController;
import com.example.demo.model.persistence.Cart;
import com.example.demo.model.persistence.Item;
import com.example.demo.model.persistence.User;
import com.example.demo.model.persistence.UserOrder;
import com.example.demo.model.persistence.repositories.OrderRepository;
import com.example.demo.model.persistence.repositories.UserRepository;
import org.junit.Before;
import org.junit.Test;
import org.springframework.http.ResponseEntity;

import java.lang.reflect.Field;
import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class OrderControllerTest {

    private OrderController orderController;

    private final UserRepository userRepository = mock(UserRepository.class);

    private final OrderRepository orderRepository = mock(OrderRepository.class);

    /**
     * Set up the test environment and inject dependencies.
     */
    @Before
    public void setUp() {
        orderController = new OrderController();
        injectDependencies(orderController, "userRepository", userRepository);
        injectDependencies(orderController, "orderRepository", orderRepository);
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

    /**
     * Tests the submitOrder method for the happy path.
     */
    @Test
    public void submitOrder() {
        User user = createUserWithCart(new BigDecimal("2.99"));
        when(userRepository.findByUsername("test")).thenReturn(user);

        final ResponseEntity<UserOrder> response = orderController.submit("test");

        assertNotNull(response);
        assertEquals(200, response.getStatusCodeValue());

        UserOrder actualUserOrder = response.getBody();
        assertNotNull(actualUserOrder);
        assertEquals(user.getCart().getItems(), actualUserOrder.getItems());
        assertEquals(user, actualUserOrder.getUser());
        assertEquals(user.getCart().getTotal(), actualUserOrder.getTotal());
    }

    /**
     * Tests the submitOrder method when the username is not found.
     */
    @Test
    public void submitOrderNotFound() {
        when(userRepository.findByUsername("test")).thenReturn(null);

        final ResponseEntity<UserOrder> response = orderController.submit("test");

        assertNotNull(response);
        assertEquals(404, response.getStatusCodeValue());
    }

    /**
     * Tests the getOrdersForUser method for the happy path.
     */
    @Test
    public void getOrdersForUser() {
        User user = createUserWithCart(new BigDecimal("2.99"));
        List<UserOrder> userOrders = createUserOrders(user);
        when(userRepository.findByUsername("test")).thenReturn(user);
        when(orderRepository.findByUser(user)).thenReturn(userOrders);

        final ResponseEntity<List<UserOrder>> response = orderController.getOrdersForUser("test");

        assertNotNull(response);
        assertEquals(200, response.getStatusCodeValue());

        List<UserOrder> actualUserOrders = response.getBody();
        assertNotNull(actualUserOrders);
        assertEquals(userOrders, actualUserOrders);
    }

    /**
     * Tests the getOrdersForUser method when the username is not found.
     */
    @Test
    public void getOrdersForUserNotFound() {
        when(userRepository.findByUsername("test")).thenReturn(null);

        final ResponseEntity<List<UserOrder>> response = orderController.getOrdersForUser("test");

        assertNotNull(response);
        assertEquals(404, response.getStatusCodeValue());
    }

    /**
     * Creates a User with a Cart filled with items.
     *
     * @param itemPrice The price of each item.
     * @return the created User
     */
    private User createUserWithCart(BigDecimal itemPrice) {
        User user = new User();
        user.setUsername("test");
        user.setPassword("password");
        user.setId(0L);

        Cart cart = new Cart();
        cart.setId(0L);
        cart.setUser(user);
        user.setCart(cart);

        Item item = new Item();
        item.setId(0L);
        item.setName("itemname");
        item.setPrice(itemPrice);
        item.setDescription("description");

        List<Item> itemsArray = new ArrayList<>();
        for (int i = 0; i < 3; i++) {
            itemsArray.add(item);
        }
        cart.setItems(itemsArray);
        cart.setTotal(itemPrice.multiply(BigDecimal.valueOf(3)));

        return user;
    }

    /**
     * Creates a list of UserOrder for a user.
     *
     * @param user The user for whom the orders are created.
     * @return list of created UserOrder
     */
    private List<UserOrder> createUserOrders(User user) {
        List<UserOrder> userOrders = new ArrayList<>();
        for (int i = 0; i < 2; i++) {
            UserOrder order = UserOrder.createFromCart(user.getCart());
            order.setId((long) i);
            userOrders.add(order);
        }
        return userOrders;
    }

}