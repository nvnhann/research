package model;

import java.util.Objects;

/**
 * Represents a room. A room has a room number, price, and room type. This class implements the
 * IRoom interface and provides methods to access room information. It also overrides the equals,
 * hashCode, and isFree methods.
 *
 * @author NhanNV13
 */
public class Room implements IRoom {
    private final String roomNumber;
    private Double price = 0.0;
    private final RoomType enumeration;

    /**
     * Constructs a Room object with the specified room number, price, and room type.
     *
     * @param roomNumber the room number
     * @param price the price of the room
     * @param enumeration the room type enumeration
     */
    public Room(final String roomNumber, Double price, final RoomType enumeration) {
        this.roomNumber = roomNumber;
        this.price = price;
        this.enumeration = enumeration;
    }

    @Override
    public String getRoomNumber() {
        return this.roomNumber;
    }

    @Override
    public Double getRoomPrice() {
        return this.price;
    }

    @Override
    public RoomType getRoomType() {
        return this.enumeration;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }

        if (obj instanceof Room) {
            Room room = (Room) obj;
            return Objects.equals(this.roomNumber, room.roomNumber);
        }

        return false;
    }

    @Override
    public int hashCode() {
        return Objects.hash(roomNumber);
    }

    @Override
    public boolean isFree() {
        return this.price.equals(0.0);
    }

    @Override
    public String toString() {
        return this.roomNumber + " - " + this.price + " - " + this.enumeration;
    }
}
