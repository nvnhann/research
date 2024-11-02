package service;

import model.Customer;
import model.IRoom;
import model.Reservation;

import java.time.LocalDate;
import java.time.ZoneId;
import java.util.*;
import java.util.stream.Collectors;

public class ReservationService {
    private static final ReservationService RESERVATION_SERVICE = new ReservationService();
    public static final int RECOMMENDED_ROOMS_DEFAULT_PLUS_DAYS = 7;
    private static HashSet<IRoom> rooms;
    private static Map<String, LinkedList<Reservation>> reservations;

    private ReservationService() {
        rooms = new HashSet<>();
        reservations = new HashMap<>();
    }

    public static ReservationService getReservationService() {
        return RESERVATION_SERVICE;
    }

    public void addRoom(IRoom room) {
        rooms.add(room);
    }

    public IRoom getARoom(String roomNumber) {
        for (IRoom room : rooms) {
            if (room.getRoomNumber().equals(roomNumber)) {
                return room;
            }
        }
        return null;
    }

    public Collection<IRoom> getAllRooms() {
        return rooms;
    }

    public Reservation reserveARoom(Customer customer, IRoom room, Date checkInDate, Date checkOutDate) {
        Reservation reservation = new Reservation(customer, room, checkInDate, checkOutDate);
        reservations.computeIfAbsent(customer.getEmail(), key -> new LinkedList<>()).add(reservation);
        return reservation;
    }

    public Collection<IRoom> findRooms(Date checkInDate, Date checkOutDate) {
        return getIRooms(checkInDate, checkOutDate);
    }

    public Collection<IRoom> findAlternativeRooms(Date checkInDate, Date checkOutDate) {
        return getIRooms(plusDate(checkInDate), plusDate(checkOutDate));
    }

    public Date plusDate(final Date date) {
        // Convert input Date to Calendar
        Calendar calendar = Calendar.getInstance();
        calendar.setTime(date);

        // Add days to the Calendar
        calendar.add(Calendar.DAY_OF_MONTH, RECOMMENDED_ROOMS_DEFAULT_PLUS_DAYS);

        // Convert the updated Calendar back to Date
        return calendar.getTime();
    }

    private Collection<IRoom> getIRooms(Date checkInDate, Date checkOutDate) {
        Collection<Reservation> allReservations = getAllReservations();
        Set<IRoom> notAvailableRooms = new HashSet<>();

        for (Reservation reservation : allReservations) {
            if (checkInDate.before(reservation.getCheckOutDate())
                    && checkOutDate.after(reservation.getCheckInDate())) {
                notAvailableRooms.add(reservation.getRoom());
            }
        }

        List<IRoom> availableRooms = new ArrayList<>();
        for (IRoom room : rooms) {
            if (!notAvailableRooms.contains(room)) {
                availableRooms.add(room);
            }
        }

        return availableRooms;
    }


    private Collection<Reservation> getAllReservations() {
        final Collection<Reservation> allReservations = new LinkedList<>();
        reservations.values().forEach(allReservations::addAll);
        return allReservations;
    }

    public Collection<Reservation> getCustomersReservation(Customer customer) {
        return reservations.get(customer.getEmail());
    }

    public void printAllReservations() {
        if (reservations.isEmpty()) {
            System.out.println("No reservations found.");
            return;
        }
        for (Collection<Reservation> customerReservations : reservations.values()) {
            for (Reservation reservation : customerReservations) {
                System.out.println(reservation);
            }
            System.out.println();
        }
    }
}