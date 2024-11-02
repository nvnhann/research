package api;

import model.Customer;
import model.IRoom;
import service.CustomerService;
import service.ReservationService;

import java.util.Collection;
import java.util.List;

/**
 * Represents the Admin Resource for managing customers and reservations. This class provides
 * methods to retrieve customers, add rooms, retrieve all rooms, display all reservations, and
 * retrieve all customers. It follows the Singleton design pattern to ensure a single instance of
 * the AdminResource.
 *
 * @author Nhannv13
 */
public class AdminResource {
    private static final AdminResource ADMINRESOURCE = new AdminResource();
    private static final CustomerService CUSTOMERSERVICE = CustomerService.getCustomerService();
    private static final ReservationService RESERVATIONSERVICE =
            ReservationService.getReservationService();

    private AdminResource() {}

    /**
     * Returns the AdminResource instance.
     *
     * @return the AdminResource instance
     */
    public static AdminResource getAdminResource() {
        return ADMINRESOURCE;
    }

    /**
     * Retrieves a customer based on the provided email.
     *
     * @param email the email address of the customer to retrieve
     * @return the Customer object associated with the email, or null if not found
     */
    public static Customer getCustomer(String email) {
        return CUSTOMERSERVICE.getCustomer(email);
    }

    /**
     * Adds a list of rooms to the reservation service.
     *
     * @param rooms the list of rooms to add
     */
    public static void addRoom(List<IRoom> rooms) {
        rooms.forEach(RESERVATIONSERVICE::addRoom);
    }

    /**
     * Retrieves all rooms from the reservation service.
     *
     * @return a collection of all available rooms
     */
    public Collection<IRoom> getAllRooms() {
        return RESERVATIONSERVICE.getAllRooms();
    }

    /**
     * Displays all reservations stored in the reservation service.
     */
    public void displayAllReservations() {
        RESERVATIONSERVICE.printAllReservations();
    }

    /**
     * Retrieves all customers from the customer service.
     *
     * @return a collection of all customers
     */
    public Collection<Customer> getAllCustomers() {
        return CUSTOMERSERVICE.getAllCustomers();
    }
    public static IRoom getRoomByRoomNumber(String roomNumber){
        return RESERVATIONSERVICE.getARoom(roomNumber);
    }
}
