import colorama
from colorama import Fore

class Knights:
    def __init__(self, board_size):
        self.whiteKnights = set()
        self.blackKnights = set()
        self.stack = []
        self.turn = "W"
        self.board_size = board_size
        self.last_move = -1

        for i in range(2):
            for j in range(board_size):
                self.blackKnights.add(j)
                self.blackKnights.add(j+board_size)
                self.whiteKnights.add(j + board_size * (board_size - 2))
                self.whiteKnights.add(j + board_size * (board_size - 1))
    
    def __str__(self):
        s = ""
        whites_color = '\033[94m'
        blacks_color = '\033[91m'
        last_move_color = '\033[92m'
        reset_color = '\033[0m'
        

        for row in range(self.board_size):
            for col in range(self.board_size):
                if row*self.board_size+col in self.whiteKnights:
                    if row*self.board_size+col == self.last_move:
                        s += last_move_color + "W " + reset_color
                    else: 
                        s += whites_color + "W " + reset_color
                elif row*self.board_size+col in self.blackKnights:
                    if row*self.board_size+col == self.last_move:
                        s += last_move_color + "B " + reset_color
                    else:
                        s += blacks_color + "B " + reset_color
                else:
                    s += "- "
            s += "\n"
        return s

    def all_actions(self):
        actions = { "W": [], "B": [] }
        deltas = [ (2, 1), (-2, 1), (2, -1), (-2, -1),
                   (1, 2), (-1, 2), (1, -2), (-1, -2) ]

        for k in self.whiteKnights:
            row, col = k // self.board_size, k % self.board_size
            for dr, dc in deltas:
                if 0 <= row+dr and row+dr < self.board_size and 0 <= col+dc and col+dc < self.board_size:
                    kto = (row+dr)*8 + col+dc
                    if kto not in self.whiteKnights:
                        actions["W"].append((k, kto))
        
        for k in self.blackKnights:
            row, col = k // self.board_size, k % self.board_size
            for dr, dc in deltas:
                if 0 <= row+dr and row+dr < self.board_size and 0 <= col+dc and col+dc < self.board_size:
                    kto = (row+dr)*self.board_size + col+dc
                    if kto not in self.blackKnights:
                        actions["B"].append((k, kto))
        
        return actions

    def make_action(self, action):
        knights = self.whiteKnights if self.turn == "W" else self.blackKnights
        kfrom, kto = action

        knights.remove(kfrom)

        if self.turn == "W" and kto in self.blackKnights:
            self.stack.append(True)
            self.blackKnights.remove(kto)
        elif self.turn == "B" and kto in self.whiteKnights:
            self.stack.append(True)
            self.whiteKnights.remove(kto)
        else:
            self.stack.append(False)
        
        knights.add(kto)

        self.change_turn()
        self.last_move = kto

    def undo_action(self, action):
        self.change_turn()
        knights = self.whiteKnights if self.turn == "W" else self.blackKnights
        kfrom, kto = action

        eaten = self.stack.pop(-1)

        knights.remove(kto)

        if eaten and self.turn == "W":
            self.blackKnights.add(kto)
        elif eaten and self.turn == "B":
            self.whiteKnights.add(kto)
        
        knights.add(kfrom)

    def game_over(self):
        if len(self.whiteKnights) >= 1 and len(self.blackKnights) == 0:
            return "W"
        elif len(self.whiteKnights) == 0 and len(self.blackKnights) >= 1:
            return "B"
        return "-"

    def change_turn(self):
        self.turn = "B" if self.turn == "W" else "W"