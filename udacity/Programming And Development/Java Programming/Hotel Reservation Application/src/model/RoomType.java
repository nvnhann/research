package model;

/**
 * Represents the type of room.
 * Each room type has a corresponding typeRoom value.
 * The typeRoom value is a string representation of the room type.
 * This enumeration provides methods to access the typeRoom value and retrieve a RoomType based on the typeRoom value.
 *
 * @author NhanNV13
 */
public enum RoomType {
    SINGLE("1"),
    DOUBLE("2");

    public final String typeRoom;

    private RoomType(String typeRoom) {
        this.typeRoom = typeRoom;
    }

    /**
     * Retrieves the RoomType based on the provided typeRoom value.
     *
     * @param typeRoom the typeRoom value to search for
     * @return the corresponding RoomType
     * @throws IllegalArgumentException if an invalid typeRoom value is provided
     */
    public static RoomType valueOfTypeRoom(String typeRoom) {
        for (RoomType roomType : values()) {
            if (roomType.typeRoom.equals(typeRoom)) {
                return roomType;
            }
        }
        throw new IllegalArgumentException("Invalid room type: " + typeRoom);
    }
}
