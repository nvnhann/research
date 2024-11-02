import api.HotelResource;
import model.IRoom;
import model.Reservation;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.*;

public class MainMenu {
    private static final Scanner scanner = new Scanner(System.in);
    private static final HotelResource HOTEL_RESOURCE = HotelResource.getHotelResource();

    public static void mainMenu() {
        int choice = -1;
        do {
            try {
                displayMainMenu();
                System.out.print("Enter your choice: ");
                choice = scanner.nextInt();
                scanner.nextLine(); // Consume the newline character
                switch (choice) {
                    case 1:
                        findAndReserveRoom();
                        continue;
                    case 2:
                        seeMyReserveRoom();
                        continue;
                    case 3:
                        createAccount();
                        continue;
                    case 4:
                        AdminMenu.adminMenu();
                        continue;
                    case 5:
                        System.out.println("Exiting the program. Goodbye!");
                        break;
                    default:
                        System.out.println("Invalid choice. Please try again.");
                }
            } catch (InputMismatchException e) {
                System.out.println("Invalid input. Please enter a valid option.");
                scanner.nextLine(); // Consume the invalid input
            }
        } while (choice != 5);
    }

    private static void displayMainMenu() {
        System.out.println("|------------------------------ MENU ------------------------------|");
        System.out.println(String.format("|    %-61s |", "1. Find and reserve a room"));
        System.out.println(String.format("|    %-61s |", "2. See my reservations"));
        System.out.println(String.format("|    %-61s |", "3. Create an account"));
        System.out.println(String.format("|    %-61s |", "4. Admin"));
        System.out.println(String.format("|    %-61s |", "5. Exit"));
        System.out.println("|------------------------------------------------------------------|");
    }

    private static void createAccount() {
        System.out.print("Enter Email (ex: name@domain.com): ");
        String email = scanner.nextLine();
        System.out.print("First Name: ");
        String fistName = scanner.nextLine();
        System.out.print("Last Name: ");
        String lastName = scanner.nextLine();
        try {
            HOTEL_RESOURCE.createACustomer(email, fistName, lastName);
            System.out.println("CREATE CUSTOMER SUCCESSFULLY!");
            System.out.println(HOTEL_RESOURCE.getCustomer(email));
        } catch (IllegalArgumentException e) {
            System.out.println(e.getMessage());
            createAccount();
        }
    }

    private static void findAndReserveRoom() {
        Date dateIn = new Date();
        Date dateOut = new Date();;
        try {
            System.out.print("Date check in (format dd/mm/yyyy ex: 19/07/2023): ");
            dateIn = inputDate();
            if (!isDateGreaterThanCurrent(dateIn)) {
                System.out.println("Invalid date!");
                findAndReserveRoom();
            }
            System.out.print("Date check out (format dd/mm/yyyy ex: 19/07/2023): ");
            dateOut = inputDate();

        } catch (ParseException e) {
            System.out.println("Date invalid!");
            findAndReserveRoom();
        }

        if (dateIn != null && dateOut != null) {
            Collection<IRoom> availableRooms = HotelResource.findRooms(dateIn, dateOut);
            if (availableRooms.isEmpty()) {
                Collection<IRoom> alternativeRooms = HOTEL_RESOURCE.findAlternativeRooms(dateIn, dateOut);
                if(alternativeRooms.isEmpty()){
                    System.out.println("Not found rooms available rooms");
                }else {
                    System.out.println("I have available room ");
                    System.out.print("From: ");
                    System.out.println(HOTEL_RESOURCE.plusDate(dateIn));
                    System.out.print("To: ");
                    System.out.println(HOTEL_RESOURCE.plusDate(dateOut));
                    displayRooms(alternativeRooms);
                    bookRoom(HOTEL_RESOURCE.plusDate(dateIn), HOTEL_RESOURCE.plusDate(dateOut), alternativeRooms);
                }
            } else {
                displayRooms(availableRooms);
                bookRoom(dateIn, dateOut, availableRooms);
            }
        }
    }

    private static void seeMyReserveRoom() {
        System.out.print("Enter Email (ex: name@domain.com): ");
        String email = scanner.nextLine();
        Collection<Reservation> customerReserves = HOTEL_RESOURCE.getCustomersReservations(email);
        if (customerReserves.isEmpty()) {
            System.out.println("Not found reservations!");
        } else {
            System.out.println(
                    "|------------------------------ RESERVATIONS ----------------------------|");
            for (Reservation reservation : customerReserves) {
                System.out.println(reservation);
            }
            System.out.println(
                    "|------------------------------------------------------------------------|");
        }
    }

    private static Date inputDate() throws ParseException {
        SimpleDateFormat dateFormat = new SimpleDateFormat("dd/MM/yyyy");
        String dateStr = scanner.nextLine();

        return dateFormat.parse(dateStr);
    }

    private static boolean isDateGreaterThanCurrent(Date date) {
        Date currentDate = new Date();
        return date.after(currentDate);
    }

    private static void displayRooms(Collection<IRoom> rooms) {
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

    private static void bookRoom(Date checkIn, Date checkOut, Collection<IRoom> rooms) {
        if (inputConfirmYesOrNo("Would you like book a room?(y/n): ")) {
            if (inputConfirmYesOrNo("Do you have an account?(y/n): ")) {
                System.out.print("Enter Email (ex: name@domain.com): ");
                String email = scanner.nextLine();
                if (HOTEL_RESOURCE.getCustomer(email) == null) {
                    System.out.println("Customer not found!");
                } else {
                    System.out.print("What number room would you like book: ");
                    String roomNumber = scanner.nextLine();
                    if (rooms.stream().anyMatch(room -> room.getRoomNumber().equals(roomNumber))) {
                        IRoom room = HOTEL_RESOURCE.getRoom(roomNumber);
                        Reservation reservation =
                                HOTEL_RESOURCE.bookARoom(email, room, checkIn, checkOut);
                        System.out.println("BOOK ROOM SUCCESSFULLY!");
                        System.out.println(reservation);
                    } else {
                        System.out.println("Room number not available!");
                    }
                }
            } else {

            }
        }
    }

    private static boolean inputConfirmYesOrNo(String message) {
        String choice;
        do {
            System.out.print(message);
            choice = scanner.nextLine();
        } while (!choice.equalsIgnoreCase("y") && !choice.equalsIgnoreCase("n"));
        return choice.equalsIgnoreCase("y");
    }

}
