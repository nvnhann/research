package com.example.demo.controller;

import com.example.demo.controllers.ItemController;
import com.example.demo.model.persistence.Item;
import com.example.demo.model.persistence.repositories.ItemRepository;
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

/**
 * Test class for ItemController.
 */
public class ItemControllerTest {

    private ItemController itemController;

    private final ItemRepository itemRepository = mock(ItemRepository.class);

    /**
     * Sets up the test dependencies.
     */
    @Before
    public void setUp() {
        itemController = new ItemController();
        injectDependencies(itemController, itemRepository);
    }

    /**
     * Injects dependencies into the target object.
     *
     * @param target   The object into which the dependency should be injected.
     * @param toInject The dependency to be injected.
     */
    private static void injectDependencies(Object target, Object toInject) {
        try {
            Field field = target.getClass().getDeclaredField("itemRepository");
            boolean accessible = field.isAccessible();
            field.setAccessible(true);
            field.set(target, toInject);
            field.setAccessible(accessible);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Tests the retrieval of all items.
     *
     * @throws Exception If any exception occurs during the test.
     */
    @Test
    public void getAll() throws Exception {
        Item item = new Item();
        item.setName("testItem");
        item.setPrice(new BigDecimal("2.99"));
        item.setDescription("This is a testItem description");

        List<Item> expectedItems = new ArrayList<>();
        for (int i = 0; i < 2; i++) {
            item.setId((long) i);
            expectedItems.add(item);
        }

        when(itemRepository.findAll()).thenReturn(expectedItems);

        final ResponseEntity<List<Item>> response = itemController.getItems();

        assertNotNull(response);
        assertEquals(200, response.getStatusCodeValue());

        List<Item> actualItems = response.getBody();
        assertNotNull(actualItems);
        assertEquals(expectedItems, actualItems);
    }

    /**
     * Tests the retrieval of an item by ID.
     *
     * @throws Exception If any exception occurs during the test.
     */
    @Test
    public void getItemById() throws Exception {
        Item item = new Item();
        item.setId((long) 0);
        item.setName("testItem");
        item.setPrice(new BigDecimal("2.99"));
        item.setDescription("This is a testItem description");

        when(itemRepository.findById((long) 0)).thenReturn(java.util.Optional.of(item));

        final ResponseEntity<Item> response = itemController.getItemById((long) 0);

        assertNotNull(response);
        assertEquals(200, response.getStatusCodeValue());

        Item actualItem = response.getBody();
        assertNotNull(actualItem);
        assertEquals(item.getId(), actualItem.getId());
        assertEquals(item.getName(), actualItem.getName());
        assertEquals(item.getPrice(), actualItem.getPrice());
        assertEquals(item.getDescription(), actualItem.getDescription());
    }

    /**
     * Tests the retrieval of items by name.
     *
     * @throws Exception If any exception occurs during the test.
     */
    @Test
    public void getItemByName() throws Exception {
        Item item = new Item();
        item.setName("testItem");
        item.setPrice(new BigDecimal("2.99"));
        item.setDescription("This is a testItem description");

        List<Item> expectedItems = new ArrayList<>();
        for (int i = 0; i < 2; i++) {
            item.setId((long) i);
            expectedItems.add(item);
        }

        when(itemRepository.findByName("testItem")).thenReturn(expectedItems);

        final ResponseEntity<List<Item>> response = itemController.getItemsByName("testItem");

        assertNotNull(response);
        assertEquals(200, response.getStatusCodeValue());

        List<Item> actualItems = response.getBody();
        assertNotNull(actualItems);
        assertEquals(expectedItems, actualItems);
    }

    /**
     * Tests the retrieval of items by name when no items are found.
     *
     * @throws Exception If any exception occurs during the test.
     */
    @Test
    public void getItemByNameEmpty() throws Exception {

        when(itemRepository.findByName("testItem")).thenReturn(null);

        final ResponseEntity<List<Item>> response = itemController.getItemsByName("testItem");

        assertNotNull(response);
        assertEquals(404, response.getStatusCodeValue());
    }

}