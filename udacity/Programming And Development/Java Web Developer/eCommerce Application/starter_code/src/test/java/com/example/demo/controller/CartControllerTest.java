package com.example.demo.controller;

import com.example.demo.controllers.CartController;
import com.example.demo.model.persistence.Cart;
import com.example.demo.model.persistence.Item;
import com.example.demo.model.persistence.User;
import com.example.demo.model.persistence.repositories.CartRepository;
import com.example.demo.model.persistence.repositories.ItemRepository;
import com.example.demo.model.persistence.repositories.UserRepository;
import com.example.demo.model.requests.ModifyCartRequest;
import org.junit.Before;
import org.junit.Test;
import org.springframework.http.ResponseEntity;

import java.lang.reflect.Field;
import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Optional;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class CartControllerTest {

    private CartController cartController;

    private UserRepository userRepository;
    private ItemRepository itemRepository;

    @Before
    public void setUp() {
        userRepository = mock(UserRepository.class);
        CartRepository cartRepository = mock(CartRepository.class);
        itemRepository = mock(ItemRepository.class);

        cartController = new CartController();
        inject(cartController, "userRepository", userRepository);
        inject(cartController, "cartRepository", cartRepository);
        inject(cartController, "itemRepository", itemRepository);
    }

    private static void inject(Object target, String fieldName, Object toInject) {
        try {
            Field field = target.getClass().getDeclaredField(fieldName);
            boolean accessible = field.isAccessible();
            field.setAccessible(true);
            field.set(target, toInject);
            field.setAccessible(accessible);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            throw new RuntimeException(e);
        }
    }

    private User createUserWithCart() {
        User user = new User();
        user.setUsername("test");
        user.setId(0);
        Cart cart = new Cart();
        cart.setId((long) 0);
        cart.setUser(user);
        user.setCart(cart);
        return user;
    }

    private Item createItem(BigDecimal price) {
        Item item = new Item();
        item.setId((long) 0);
        item.setName("testItem");
        item.setPrice(price);
        item.setDescription("This is a testItem description");
        return item;
    }

    @Test
    public void addCart() {
        User user = createUserWithCart();
        Item item = createItem(new BigDecimal("2.99"));

        ModifyCartRequest modifyCartRequest = new ModifyCartRequest();
        modifyCartRequest.setUsername("test");
        modifyCartRequest.setItemId(0);
        modifyCartRequest.setQuantity(3);

        when(userRepository.findByUsername("test")).thenReturn(user);
        when(itemRepository.findById(0L)).thenReturn(Optional.of(item));

        ResponseEntity<Cart> response = cartController.addTocart(modifyCartRequest);

        assertNotNull(response);
        assertEquals(200, response.getStatusCodeValue());

        Cart actualCart = response.getBody();
        assertNotNull(actualCart);
        assertEquals(user.getCart().getId(), actualCart.getId());

        List<Item> expectedItems = Collections.nCopies(modifyCartRequest.getQuantity(), item);
        assertEquals(expectedItems, actualCart.getItems());
        assertEquals(user, actualCart.getUser());
        assertEquals(new BigDecimal("8.97"), actualCart.getTotal());
    }

    @Test
    public void addCartNotFound() {
        ModifyCartRequest modifyCartRequest = new ModifyCartRequest();
        modifyCartRequest.setUsername("test");
        modifyCartRequest.setItemId(0);
        modifyCartRequest.setQuantity(3);

        when(userRepository.findByUsername("test")).thenReturn(null);

        ResponseEntity<Cart> response = cartController.addTocart(modifyCartRequest);

        assertNotNull(response);
        assertEquals(404, response.getStatusCodeValue());
    }

    @Test
    public void addCartItemNotFound() {
        User user = createUserWithCart();

        ModifyCartRequest modifyCartRequest = new ModifyCartRequest();
        modifyCartRequest.setUsername("test");
        modifyCartRequest.setItemId(1);
        modifyCartRequest.setQuantity(3);

        when(userRepository.findByUsername("test")).thenReturn(user);
        when(itemRepository.findById(1L)).thenReturn(Optional.empty());

        ResponseEntity<Cart> response = cartController.addTocart(modifyCartRequest);

        assertNotNull(response);
        assertEquals(404, response.getStatusCodeValue());
    }

    @Test
    public void removeCart() {
        User user = createUserWithCart();
        Item item = createItem(BigDecimal.valueOf(2.99));

        ModifyCartRequest modifyCartRequest = new ModifyCartRequest();
        modifyCartRequest.setUsername("test");
        modifyCartRequest.setItemId(0);
        modifyCartRequest.setQuantity(1);

        List<Item> itemsArray = new ArrayList<>(Collections.nCopies(3, item));
        user.getCart().setItems(itemsArray);
        user.getCart().setTotal(BigDecimal.valueOf(8.97));

        when(userRepository.findByUsername("test")).thenReturn(user);
        when(itemRepository.findById(0L)).thenReturn(Optional.of(item));

        ResponseEntity<Cart> response = cartController.removeFromcart(modifyCartRequest);

        assertNotNull(response);
        assertEquals(200, response.getStatusCodeValue());

        Cart actualCart = response.getBody();
        assertNotNull(actualCart);
        assertEquals(user.getCart().getId(), actualCart.getId());

        List<Item> expectedItemsArray = new ArrayList<>(Collections.nCopies(2, item));
        assertEquals(expectedItemsArray, actualCart.getItems());
        assertEquals(user, actualCart.getUser());
        assertEquals(BigDecimal.valueOf(5.98), actualCart.getTotal());
    }

    @Test
    public void removeCartNotFound() {
        ModifyCartRequest modifyCartRequest = new ModifyCartRequest();
        modifyCartRequest.setUsername("test");
        modifyCartRequest.setItemId(0);
        modifyCartRequest.setQuantity(1);

        when(userRepository.findByUsername("test")).thenReturn(null);

        ResponseEntity<Cart> response = cartController.removeFromcart(modifyCartRequest);

        assertNotNull(response);
        assertEquals(404, response.getStatusCodeValue());
    }

    @Test
    public void removeCartItemNotFound() {
        User user = createUserWithCart();

        ModifyCartRequest modifyCartRequest = new ModifyCartRequest();
        modifyCartRequest.setUsername("test");
        modifyCartRequest.setItemId(1);
        modifyCartRequest.setQuantity(1);

        when(userRepository.findByUsername("test")).thenReturn(user);
        when(itemRepository.findById(1L)).thenReturn(Optional.empty());

        ResponseEntity<Cart> response = cartController.removeFromcart(modifyCartRequest);

        assertNotNull(response);
        assertEquals(404, response.getStatusCodeValue());
    }
}