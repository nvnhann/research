package model;

/**
 * Represents an interface for a room.
 * A room has a room number, room price, room type, and availability status.
 * This interface defines methods to retrieve room information.
 *
 * @author NhanNV13
 */
public interface IRoom {
    /**
     * Returns the room number.
     *
     * @return the room number
     */
    public String getRoomNumber();

    /**
     * Returns the room price.
     *
     * @return the room price
     */
    public Double getRoomPrice();

    /**
     * Returns the room type.
     *
     * @return the room type
     */
    public RoomType getRoomType();

    /**
     * Checks if the room is available (free) or occupied.
     *
     * @return true if the room is free, false if occupied
     */
    public boolean isFree();
}
