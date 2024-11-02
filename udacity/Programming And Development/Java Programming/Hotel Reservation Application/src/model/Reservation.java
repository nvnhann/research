package model;

import java.util.Date;

/**
 * Represents a reservation made by a customer for a room. A reservation includes information about
 * the customer, room, check-in date, and check-out date. This class provides methods to access the
 * reservation details and overrides the toString method.
 *
 * @author NhanNV13
 */
public class Reservation {
    private final Customer customer;
    private final IRoom room;
    private final Date checkInDate;
    private final Date checkOutDate;

    /**
     * Constructs a Reservation object with the specified customer, room, check-in date, and
     * check-out date.
     *
     * @param customer     the customer making the reservation
     * @param room         the room reserved
     * @param checkInDate  the check-in date
     * @param checkOutDate the check-out date
     */
    public Reservation(Customer customer, IRoom room, Date checkInDate, Date checkOutDate) {
        this.customer = customer;
        this.room = room;
        this.checkInDate = checkInDate;
        this.checkOutDate = checkOutDate;
    }

    /**
     * Returns the room reserved in the reservation.
     *
     * @return the reserved room
     */
    public IRoom getRoom() {
        return this.room;
    }

    public Customer getCustomer() {
        return this.customer;
    }
    /**
     * Returns the check-in date of the reservation.
     *
     * @return the check-in date
     */
    public Date getCheckInDate() {
        return this.checkInDate;
    }

    /**
     * Returns the check-out date of the reservation.
     *
     * @return the check-out date
     */
    public Date getCheckOutDate() {
        return this.checkOutDate;
    }

    @Override
    public String toString() {
        return "CUSTOMER:: " + this.customer.toString() + "\nROOM:: " + this.room.toString()
                + "\nCHECKIN DATE:: " + this.checkInDate + "\nCHECKOUT DATE:: " + this.checkOutDate;
    }
}
