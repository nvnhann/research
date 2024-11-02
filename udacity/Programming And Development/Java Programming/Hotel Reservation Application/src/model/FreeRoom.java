package model;

/**
 * Represents a free room, which is a type of room with no cost.
 * A free room inherits from the Room class and overrides the toString method.
 * This class provides a constructor to create a free room and format its string representation.
 * It includes information about the author and the parameters used in the constructor.
 *
 * @author NhanNV13
 */
public class FreeRoom extends Room {
    /**
     * Constructs a FreeRoom object with the specified room number and room type.
     * The price is set to 0.0 to indicate that it is a free room.
     *
     * @param roomNumber  the room number
     * @param enumeration the room type enumeration
     */
    public FreeRoom(String roomNumber, RoomType enumeration) {
        super(roomNumber, 0.0, enumeration);
    }

    @Override
    public String toString() {
        return "========================= FREE ROOM ========================= \n" +
                super.toString() +
                "============================================================= \n";
    }
}
