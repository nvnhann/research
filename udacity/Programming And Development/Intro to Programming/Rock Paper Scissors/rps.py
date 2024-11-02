#!/usr/bin/env python3
import random
"""This program plays a game of Rock, Paper, Scissors between two Players,
and reports both Player's scores each round."""

moves = ['rock', 'paper', 'scissors']

"""The Player class is the parent class for all of the Players
in this game"""


# class layer
class Player:
    def __init__(self) -> None:
        self.my_move = random.choice(moves)
        self.their_move = random.choice(moves)

    def move(self, move):
        self.my_move = move

# method learn
    def learn(self, my_move, their_move):
        self.my_move = my_move
        self.their_move = their_move


# class human
class Human(Player):
    def __init__(self):
        super().__init__()

    def move(self):
        move = input_choice("Answer(rock/paper/scissors): ", moves)
        super().move(move)


# A player that chooses its moves randomly
class PlayerRandom(Player):
    def __init__(self):
        super().__init__()

    def move(self):
        super().move(random.choice(moves))


#  A player that always plays 'rock'
class PlayerRock(Player):
    def __init__(self):
        super().__init__()

    def move(self):
        super().move('rock')


# A player remembers and imitates what the human player in the previous round.
class PlayerImitate(Player):
    def __init__(self):
        super().__init__()

    def move(self):
        super().move(self.their_move)


# A player that cycles through the three moves
class PlayerCycle(Player):
    def __init__(self):
        super().__init__()

    def move(self):
        move = moves[(moves.index(self.my_move) + 1) % len(moves)]
        super().move(move)


def beats(one, two):
    if one == two:
        return None
    return ((one == 'rock' and two == 'scissors') or
            (one == 'scissors' and two == 'paper') or
            (one == 'paper' and two == 'rock'))


def get_score(result):
    if result is None:
        return "Draw"
    elif result:
        return "Win"
    else:
        return "Loss"


def input_choice(msg, options):
    while True:
        choice = input(msg)
        for option in options:
            if option in choice:
                return option
        print("That is not a valid option.")


class Game:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2
        self.score1 = 0
        self.score2 = 0

    def play_round(self):
        self.p1.move()
        print(self.p2.their_move)
        self.p2.move()
        result = beats(self.p1.my_move, self.p2.my_move)
        print(f"Player 1: {self.p1.my_move}  Player 2: {self.p2.my_move}  "
              f"Score: Player 1 - {get_score(result)}")
        self.p1.learn(self.p1.my_move, self.p2.my_move)
        self.p2.learn(self.p2.my_move, self.p1.my_move)
        self.score1 += 1 if result else 0
        self.score2 += 1 if result is False else 0

    def play_game(self):
        for round in range(3):
            print(f"Round {round + 1}:")
            self.play_round()
        print("Game over!")

    def result(self):
        if self.score1 > self.score2:
            return "Player 1 win."
        elif self.score1 < self.score2:
            return "Player 2 win."
        else:
            return "It's a draw"


def input_playgame():
    while True:
        choice = input("Answer (rock/scissors/paper): ")
        for option in moves:
            if option in choice:
                return option
        print("That is not a valid option.")


def main():
    print("ROCK - PAPER - SCISSORS")
    person = Human()
    computer = random.choice(
        [PlayerImitate(), PlayerRandom(), PlayerCycle(), PlayerRock()]
        )
    game = Game(person, computer)
    game.play_game()
    print(game.result())


if __name__ == "__main__":
    main()
