#  Game Code
import random
import time


#  Function that prints out descriptions
def adventure_desc(description):
    for char in description:
        print(char, end='', flush=True)
        time.sleep(0.025)
    print()


#  Function that asks user for input
def get_choice(msg, options):
    while True:
        adventure_desc(msg)
        choice = input("Answer: ")
        for option in options:
            if option in choice:
                return option
        print("That is not a valid option.")


#  Function that generates a random choice from a list
def random_choice(list_of_choices):
    return random.choice(list_of_choices)


#  knowledge person
def knowledge_person():
    list_person = [
        f"Leonardo da Vinci: Leonardo da Vinci, the famous Renaissance"
        f"polymath as not only an accomplished painter but"
        f"also an engineer, scientist, and inventor.\n"
        f"His sketchbooks contain designs for flying machines,"
        f"armored vehicles, and advanced weaponry, \n"
        f"showcasing his incredible imagination and"
        f"forward-thinking ideas.",
        f"Marie Curie: Marie Curie, a pioneering physicist and chemist,"
        f"was the first woman to win a Nobel Prize and remains the only person"
        f"to have received Nobel Prizes in two different scientific"
        f"fields (Physics and Chemistry)"
        f".Her groundbreaking research on radioactivity laid the"
        f"foundation for modern nuclear physics and medicine.",
        f"Amelia Earhart: Amelia Earhart was an aviation trailblazer"
        f"and the first female aviator to fly solo across the Atlantic Ocean\n"
        f"She disappeared mysteriously in 1937 while attempting to"
        f"circumnavigate empowerment and courage in the face of challenges.\n",
        f"Nelson Mandela: Nelson Mandela was a key figure in the fight against"
        f"apartheid in South Africa. After spending 27 years in prison,\n"
        f"he emerged as a unifying force, promoting forgiveness"
        f"and reconciliation between South Africa's racial groups.\n"
        f"He became the country's first black president in 1994,"
        f"bringing an end to decades of racial segregation."
    ]
    print("=============================="
          f"============================="
          f" PERSON "
          f"=============================="
          f"=============================")
    person = random_choice(list_person)
    adventure_desc(person)


def knowledge_animal():
    list_animal = [
        f"Lion: The lion is a majestic big cat known for its"
        f"powerful roar and regal appearance.\n"
        f"It is a social animal, living in prides led by dominant males.\n"
        f"Lions are skilled hunters, primarily preying on large herbivores.\n"
        f"They are often seen as symbols of strength "
        f"and courage in various cultures.",
        f"Dolphin: Dolphins are highly intelligent marine mammals,"
        f"known for their playful behavior and strong social bonds.\n"
        f"They use echolocation to navigate and communicate,"
        f"making them excellent hunters and communicators.\n"
        f"Dolphins are beloved by many for their friendly "
        f"and interactive nature.",
        f"Eagle: The eagle is a powerful bird of prey,"
        f"with keen eyesight and strong flying abilities.\n"
        f"It is often associated with freedom and vision,"
        f"and its majestic appearance has made it a symbol "
        f"of power in various cultures.\n"
        f"Eagles are skilled hunters and can be found on "
        f"every continent except Antarctica.",
        f"Elephant: Elephants are the largest land animals,"
        f"known for their long trunks, large ears, and incredible memory.\n"
        f"They live in tight-knit family groups and "
        f"demonstrate strong emotions,"
        f"including grief and joy. Elephants are herbivores "
        f"and play a vital role"
        f"in shaping their ecosystems by creating water holes "
        f"and clearing vegetation."
    ]
    print("=============================="
          f"============================="
          f" ANIMAL "
          f"=============================="
          f"=============================")
    person = random_choice(list_animal)
    adventure_desc(person)


#  intro game
def intro():
    adventure_desc("Welcome to the adventure game!")
    adventure_desc("The game will give you many interesting things!")
    adventure_desc("You will choose a person or an animal the game will "
                   f"provide interesting knowledge about the thing you choose")


#  The main game loop
def start_game():
    intro()
    while True:
        options = ["person", "animal"]
        option_choice = get_choice("Do you choose person or animal?", options)
        if option_choice == "person":
            knowledge_person()
        if option_choice == "animal":
            knowledge_animal()
        options = ["yes", "no"]
        option_choice = get_choice("Do you want to play again?", options)
        if option_choice == "yes":
            continue
        elif option_choice == "no":
            adventure_desc("It's time to say goodbye!!.")
            break


if __name__ == "__main__":
    start_game()
