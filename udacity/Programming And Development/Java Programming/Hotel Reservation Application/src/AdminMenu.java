import java.util.Collection;
import java.util.Collections;
import java.util.Scanner;

import api.AdminResource;
import model.Customer;
import model.IRoom;
import model.Room;
import model.RoomType;

public class AdminMenu {
    private static final Scanner scanner = new Scanner(System.in);
    private static final AdminResource ADMIN_RESOURCE = AdminResource.getAdminResource();

    public static void adminMenu() {
        int choice = -1;
        do {
            displayMenu();
            System.out.print("Enter your choice: ");
            choice = scanner.nextInt();
            scanner.nextLine(); // Consume the newline character
            switch (choice) {
                case 1:
                    displayAllCustomer();
                    continue;
                case 2:
                    displayAllRoom();
                    continue;
                case 3:
                    displayAllReservations();
                    continue;
                case 4:
                    addRoom();
                    continue;
                case 5:
                    break;
                default:
                    System.out.println("Invalid choice. Please try again.");
            }
        } while (choice != 5);
    }

    private static void displayMenu() {
        System.out.println(
                "|------------------------------ ADMIN MENU ------------------------------|");
        System.out.println(String.format("|    %-67s |", "1. See all Customers"));
        System.out.println(String.format("|    %-67s |", "2. See all Rooms"));
        System.out.println(String.format("|    %-67s |", "3. See all Reservations"));
        System.out.println(String.format("|    %-67s |", "4. Add a Room"));
        System.out.println(String.format("|    %-67s |", "5. Back to Main Menu"));
        System.out.println(
                "|------------------------------------------------------------------------|");
    }

    private static void displayAllCustomer() {
        Collection<Customer> customers = ADMIN_RESOURCE.getAllCustomers();
        if (customers.isEmpty()) {
            System.out.println("No rooms found.");
        } else {
            int i = 1;
            System.out.println(
                    "|------------------------------- CUSTOMERS ------------------------------|");
            for (Customer customer : customers) {
                System.out.println(String.format("|    %-67s |", i + ": " + customer));
                i++;
            }
            System.out.println(
                    "|------------------------------------------------------------------------|");
        }

    }

    private static void displayAllRoom() {
        Collection<IRoom> rooms = ADMIN_RESOURCE.getAllRooms();

        if (rooms.isEmpty()) {
            System.out.println("No rooms found.");
        } else {
            int i = 1;
            System.out.println(
                    "|--------------------------------- ROOMS --------------------------------|");
            for (IRoom room : rooms) {
                System.out.println(String.format("|    %-67s |", i + ": " + room));
                i++;
            }
            System.out.println(
                    "|------------------------------------------------------------------------|");
        }
    }

    private static void displayAllReservations() {
        ADMIN_RESOURCE.displayAllReservations();
    }

    private static void addRoom() {
        String roomNumber = getInputRoomNumber();
        double roomPrice = getPositiveDoubleInput("Room price: ");
        RoomType roomType = roomType();

        Room room = new Room(roomNumber, roomPrice, roomType);
        AdminResource.addRoom(Collections.singletonList(room));
        System.out.println("ADD ROOM SUCCESSFULLY!");
        System.out.println(room);
        if (shouldAddAnotherRoom()) {
            addRoom();
        }
    }

    private static String getUserInput(String prompt) {
        System.out.print(prompt);
        return scanner.nextLine();
    }

    private static String getInputRoomNumber() {
        Scanner scanner = new Scanner(System.in);

        while (true) {
            System.out.print("Room number: ");
            String roomNumber = scanner.nextLine();

            if (!roomNumber.matches(".*\\d.*")) {
                System.out.println("Invalid room number! Please enter a least one digit.");
            } else {

                    return roomNumber;
            }
        }
    }


    private static double getPositiveDoubleInput(String prompt) {
        String priceNumber;
        do {
            priceNumber = getUserInput(prompt);
            if (!isPositiveDouble(priceNumber)) {
                System.out.println("Invalid room price!");
            }
        } while (!isPositiveDouble(priceNumber));
        return Double.parseDouble(priceNumber);
    }

    private static boolean isPositiveDouble(String input) {
        try {
            double number = Double.parseDouble(input);
            return number > 0;
        } catch (NumberFormatException e) {
            return false;
        }
    }

    private static RoomType roomType() {
        try {
            System.out.print("Room type (1: single, 2: double): ");
            return RoomType.valueOfTypeRoom(scanner.nextLine());
        } catch (IllegalArgumentException E) {
            System.out.println("Invalid room type!");
            return roomType();
        }
    }

    private static boolean shouldAddAnotherRoom() {
        String choice;
        do {
            choice = getUserInput("Do you want to add another room? (y/n): ");
        } while (!choice.equalsIgnoreCase("y") && !choice.equalsIgnoreCase("n"));
        return choice.equalsIgnoreCase("y");
    }

}
