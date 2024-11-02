package api;

import model.Customer;
import model.IRoom;
import model.Reservation;
import service.CustomerService;
import service.ReservationService;

import java.util.Collection;
import java.util.Collections;
import java.util.Date;

/**
 * Represents the Hotel Resource for managing customers, rooms, and reservations. This class
 * provides methods to retrieve customers, create a new customer, retrieve rooms, book a room,
 * retrieve a customer's reservations, and find available rooms for booking. It follows the
 * Singleton design pattern to ensure a single instance of the HotelResource. The author information
 * is included in the docstring.
 *
 * @author Nhannv13
 */
public class HotelResource {
    private static final HotelResource HOTEL_RESOURCE = new HotelResource();
    private static final CustomerService CUSTOMER_SERVICE = CustomerService.getCustomerService();
    private static final ReservationService RESERVATION_SERVICE =
            ReservationService.getReservationService();

    /**
     * Returns the HotelResource instance.
     *
     * @return the HotelResource instance
     */
    public static HotelResource getHotelResource() {
        return HOTEL_RESOURCE;
    }

    /**
     * Retrieves a customer based on the provided email.
     *
     * @param email the email address of the customer to retrieve
     * @return the Customer object associated with the email, or null if not found
     */
    public Customer getCustomer(String email) {
        return CUSTOMER_SERVICE.getCustomer(email);
    }

    /**
     * Creates a new customer with the provided email, first name, and last name.
     *
     * @param email the email address of the new customer
     * @param firstName the first name of the new customer
     * @param lastName the last name of the new customer
     */
    public void createACustomer(String email, String firstName, String lastName) {
        CUSTOMER_SERVICE.addCustomer(email, firstName, lastName);
    }

    /**
     * Retrieves a room based on the provided room number.
     *
     * @param roomNumber the room number to search for
     * @return the IRoom object associated with the room number, or null if not found
     */
    public IRoom getRoom(String roomNumber) {
        return RESERVATION_SERVICE.getARoom(roomNumber);
    }

    /**
     * Books a room for a customer with the specified check-in and check-out dates.
     *
     * @param customerEmail the email address of the customer making the reservation
     * @param room the room to reserve
     * @param checkInDate the check-in date
     * @param checkOutDate the check-out date
     * @return the Reservation object representing the reservation made
     */
    public Reservation bookARoom(String customerEmail, IRoom room, Date checkInDate,
            Date checkOutDate) {
        Customer customer = getCustomer(customerEmail);
        return RESERVATION_SERVICE.reserveARoom(customer, room, checkInDate, checkOutDate);
    }

    /**
     * Retrieves all reservations made by a specific customer.
     *
     * @param customerEmail the email address of the customer
     * @return a collection of Reservation objects representing the customer's reservations
     */
    public Collection<Reservation> getCustomersReservations(String customerEmail) {
        Customer customer = getCustomer(customerEmail);
        return customer != null ? RESERVATION_SERVICE.getCustomersReservation(customer)
                : Collections.emptyList();
    }

    /**
     * Finds available rooms for booking within the specified check-in and check-out dates.
     *
     * @param checkIn the check-in date
     * @param checkOut the check-out date
     * @return a collection of available IRoom objects for the specified date range
     */
    public static Collection<IRoom> findRooms(Date checkIn, Date checkOut) {
        return RESERVATION_SERVICE.findRooms(checkIn, checkOut);
    }


    public static Collection<IRoom> findAlternativeRooms (Date checkIn, Date checkOut) {
        return RESERVATION_SERVICE.findAlternativeRooms(checkIn, checkOut);
    }

    public static Date plusDate(Date date){
        return RESERVATION_SERVICE.plusDate(date);
    }
}
