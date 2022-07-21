from Knights import Knights
from typing import Tuple
from enum import Enum
import time
import random
import numpy as np

class Mode(Enum):
    Agressive = {
                    "positioning": 4,
                    "connectivity": 5,
                    "mobility": 3,
                    "tempo": 2,
                    "material": 80,
                    "threat": 4
                }

class TomasPlayer():
    POSITIONING_POINTS =    [   
                                2, 3, 4, 4, 4, 4, 3, 2,
                                3, 4, 6, 6, 6, 6, 4, 3,
                                4, 6, 8, 8, 8, 8, 6, 4,
                                4, 6, 8, 8, 8, 8, 6, 4,
                                4, 6, 8, 8, 8, 8, 6, 4,
                                4, 6, 8, 8, 8, 8, 6, 4,
                                3, 4, 6, 6, 6, 6, 4, 3,
                                2, 3, 4, 4, 4, 4, 3, 2
                            ]
    INFINITY = 10000000
    ALPHA_START = -1000000
    BETA_START = 1000000
    WIN = 100000
    LOSE = -100000
    HASH_TABLE_SIZE = 2^30 - 1
    MAX_DEPTH = 50
    NODES_VISITED = 0

    def __init__(self, game: Knights, color: str, board_size: int, mode: Mode, time_per_turn=3) -> None:
        self.time_per_turn = time_per_turn
        self.color = color
        self.mode = mode
        self.board_size = board_size
        self.zob_piece_table = np.zeros((self.board_size, self.board_size, 2))
        self.generate_zobrist_numbers()
        self.init_hash(game)
        self.hash_table = {}
        self.killer_moves = [[(0, 0) for _ in range(self.MAX_DEPTH)] for _ in range(2)]
        self.history_moves = [[0 for _ in range(self.board_size * self.board_size)] for _ in range(self.board_size * self.board_size)]
    
    def init_hash(self, game: Knights) -> None:
        self.current_hash = 0

        for knight in game.whiteKnights:
            row, col = knight // game.board_size, knight % game.board_size
            zobrist_value = self.zob_piece_table[row][col][0]
            self.current_hash ^= np.int64(zobrist_value)
        
        for knight in game.blackKnights:
            row, col = knight // game.board_size, knight % game.board_size
            zobrist_value = self.zob_piece_table[row][col][1]
            self.current_hash ^= np.int64(zobrist_value)

        if game.turn == "B":
            self.current_hash ^= self.zob_is_black_turn


    def generate_zobrist_numbers(self):
        for i in range(self.board_size):
            for j in range(self.board_size):
                for k in range(2):
                    self.zob_piece_table[i][j][k] = random.randint(1, 2**64 - 1)
        
        self.zob_is_black_turn  = random.randint(0, 1)
    
    def store_in_hash_table(self, depth: int, score: int, best_move: Tuple[int, int]):
        key = self.current_hash & self.HASH_TABLE_SIZE

        if key not in self.hash_table:
            self.hash_table[key] = { "zob_key": self.current_hash, "depth": depth, "score": score, "best_move": best_move }    
        if depth > self.hash_table[key]["depth"]:
            self.hash_table[key] = { "zob_key": self.current_hash, "depth": depth, "score": score, "best_move": best_move }
        if depth == self.hash_table[key]["depth"] and best_move != None:
            self.hash_table[key] = { "zob_key": self.current_hash, "depth": depth, "score": score, "best_move": best_move }
        
    def get_from_hash_table(self):
        key = self.current_hash & self.HASH_TABLE_SIZE

        if key in self.hash_table and self.hash_table[key]["zob_key"] == self.current_hash:
            return self.hash_table[key]
        
        return None

    def make_action(self, game: Knights, move: Tuple[int, int]) -> None:
        kfrom, kto = move
        from_row, from_col = kfrom // game.board_size, kfrom % game.board_size
        to_row, to_col = kto // game.board_size, kto % game.board_size
        turn = 0 if game.turn == 'W' else 1
        
        zobrist_value = self.zob_piece_table[from_row][from_col][turn]
        self.current_hash ^= np.int64(zobrist_value)

        zobrist_value = self.zob_piece_table[to_row][to_col][turn]
        self.current_hash ^= np.int64(zobrist_value)
        
        game.make_action(move)
    
    def undo_action(self, game: Knights, move: Tuple[int, int]) -> None:
        kfrom, kto = move
        from_row, from_col = kfrom // game.board_size, kfrom % game.board_size
        to_row, to_col = kto // game.board_size, kto % game.board_size
        turn = 0 if game.turn == 'W' else 1
        
        zobrist_value = self.zob_piece_table[from_row][from_col][turn]
        self.current_hash ^= np.int64(zobrist_value)

        zobrist_value = self.zob_piece_table[to_row][to_col][turn]
        self.current_hash ^= np.int64(zobrist_value)
        
        game.undo_action(move)


    def get_move(self, game: Knights):
        start_time = time.time()
        max_time = start_time + self.time_per_turn

        best_score = -self.INFINITY
        best_move = None
        self.current_depth = 0
        
        sorted_moves = self.sort_moves(game, game.all_actions()[self.color])

        while time.time() < max_time:
            curr_best_score = -self.INFINITY
            curr_best_move = None
            
            for move in sorted_moves:
                self.make_action(game, move)

                local_score = self.minimax(game, self.current_depth, self.ALPHA_START, self.BETA_START)

                if local_score == self.WIN:
                    return move

                if local_score > curr_best_score:
                    curr_best_score = local_score
                    curr_best_move = move

                self.undo_action(game, move)

                if self.NODES_VISITED % 1000 == 0:
                    if time.time() >= max_time and self.current_depth > 1:
                        self.store_in_hash_table(self.current_depth, best_score, best_move)
                        print("Picked move: ", best_move)
                        print("Nodes per second: ", self.NODES_VISITED / (time.time() - start_time))
                        print("Max depth visited: ", self.current_depth - 1)
                        self.NODES_VISITED = 0
                        return best_move

            self.current_depth += 1
            best_move = curr_best_move
            best_score = curr_best_score

        self.store_in_hash_table(self.current_depth, best_score, best_move)
        print("Picked move: " + str(best_move))
        print("Nodes per second: ", self.NODES_VISITED / (time.time() - start_time))
        print("Max depth visited: " + str(self.current_depth))
        self.NODES_VISITED = 0
        return best_move

    def minimax(self, game: Knights, depth: int, alpha: int, beta: int):
        self.NODES_VISITED += 1

        if game.game_over() != '-':
            if game.game_over() == self.color:
                return self.WIN
            else:
                return self.LOSE
            
        if depth == 0:
            score = self.quiescence(game, 3, alpha, beta)
            return score
        
        table_entry = self.get_from_hash_table() 
        
        # check if current board state is in hash table
        if table_entry and table_entry["depth"] >= depth:
            return table_entry["score"]
            
        if game.turn == self.color:
            best_move = None

            sorted_moves = self.sort_moves(game, game.all_actions()[self.color])
            for move in sorted_moves:
                self.make_action(game, move)
                
                action_value = self.minimax(game, depth - 1, alpha, beta)
                self.store_in_hash_table(depth, action_value, None)

                if action_value > alpha:
                    best_move = move

                alpha = max(action_value, alpha)

                self.undo_action(game, move)

                # si la mejor opcion para A es mejor que la mejor opcion para B, salir del loop (el jugador B nunca irá por ese camino)
                if alpha >= beta and not self.is_capture(game, move):
                    self.store_killer_move(best_move, self.current_depth - depth)
                    self.history_moves[move[0]][move[1]] += 1
                    break

            self.store_in_hash_table(depth, alpha, best_move)
            return alpha
        else:
            best_move = None

            opp_color = "W" if self.color == "B" else "B"
            sorted_moves = self.sort_moves(game, game.all_actions()[opp_color])
            for move in sorted_moves:
                self.make_action(game, move)
                
                action_value = self.minimax(game, depth - 1, alpha, beta)
                self.store_in_hash_table(depth, action_value, None)

                if action_value < beta:
                    best_move = move

                self.undo_action(game, move)

                beta = min(action_value, beta)

                # si la mejor opcion para A es mejor que la mejor opcion para B, salir del loop (el jugador B nunca irá por ese camino)
                if alpha >= beta and not self.is_capture(game, move):
                    self.store_killer_move(best_move, self.current_depth - depth)
                    self.history_moves[move[0]][move[1]] += 1
                    break
            
            self.store_in_hash_table(depth, beta, best_move)
            return beta

    def quiescence(self, game: Knights, depth, alpha: int, beta: int):
        score = self.score(game)

        if score == self.WIN or depth == 0:
            return score
        
        if score >= beta:
            return beta
        if alpha < score:
            alpha = score

        if game.turn == self.color:
            sorted_moves = self.sort_moves(game, game.all_actions()[self.color])
            for move in sorted_moves:
                if self.is_capture(game, move):
                    self.make_action(game, move)        
                    alpha = self.quiescence(game, depth - 1, alpha, beta)
                    self.store_in_hash_table(self.current_depth + depth, alpha, None)
                    self.undo_action(game, move)

                    if alpha >= beta:
                        break
            return alpha
        else:
            opp_color = "W" if self.color == "B" else "B"
            sorted_moves = self.sort_moves(game, game.all_actions()[opp_color])
            for move in sorted_moves:
                if self.is_capture(game, move):
                    self.make_action(game, move)        
                    beta = self.quiescence(game, depth - 1, alpha, beta)
                    self.store_in_hash_table(self.current_depth + depth, beta, None)
                    self.undo_action(game, move)

                    if alpha >= beta:
                        break
            return beta
    
    def store_killer_move(self, killer_move: Tuple[int, int], depth: int):
        if self.killer_moves[0][depth] != killer_move:
            self.killer_moves[1][depth] = self.killer_moves[0][depth]
            self.killer_moves[0][depth] = killer_move

    def sort_moves(self, game: Knights, all_moves: list):
        def orderer(move):
            return self.evaluate_move(game, move)

        in_order = sorted(all_moves, key=orderer, reverse=True)
        return list(in_order)

    def evaluate_move(self, game: Knights, move: Tuple[int, int]):
        table_entry = self.get_from_hash_table()
        
        if table_entry and table_entry["best_move"]:
            return 10000
        
        if move == self.killer_moves[0][self.current_depth] or move == self.killer_moves[1][self.current_depth]:
            return 9000
        
        if self.is_capture(game, move):
            return 8000
        
        kfrom, kto = move
        return self.history_moves[kfrom][kto]
        

    def is_capture(self, game: Knights, move: Tuple[int, int]):
        opp_knights = game.blackKnights if game.turn == "W" else game.whiteKnights
        _, kto = move
        if kto in opp_knights:
            return True

        return False
    
    def is_defending(self, game: Knights, knight: Tuple[int, int], color: str):
        mine = game.whiteKnights if color == "W" else game.blackKnights
        deltas = [ (2, 1), (-2, 1), (2, -1), (-2, -1),
                   (1, 2), (-1, 2), (1, -2), (-1, -2) ]

        row, col = knight // self.board_size, knight % self.board_size
        for dr, dc in deltas:
            kto = (row+dr)*8 + col+dc
            if kto in mine:
                return True
        
        return False
        
    def score(self, game: Knights):
        if game.game_over() != '-':
            if game.game_over() == self.color:
                return self.WIN
            else:
                return self.LOSE

        material = self.material(game)
        positioning = self.positioning(game)
        threat = -self.threat(game) if game.turn != self.color else self.threat(game)
        connectivity = self.connectivity(game)
        mobility = self.mobility(game)
        tempo = self.tempo(game)

        return self.mode.value["material"] * material + \
                self.mode.value["positioning"] * positioning + \
                self.mode.value["threat"] * threat + \
                self.mode.value["connectivity"] * connectivity + \
                self.mode.value["mobility"] * mobility + \
                self.mode.value["tempo"] * tempo

    def positioning(self, game: Knights):
        mine, opp = (game.whiteKnights, game.blackKnights) if self.color == "W" else (game.blackKnights, game.whiteKnights)
        
        score = 0
        for knight in mine:
            score += self.POSITIONING_POINTS[knight]
        
        for knight in opp:
            score -= self.POSITIONING_POINTS[knight]
        
        return score
    
    def connectivity(self, game: Knights):
        mine, opp = (game.whiteKnights, game.blackKnights) if self.color == "W" else (game.blackKnights, game.whiteKnights)

        score = 0
        for knight in mine:
            if self.is_defending(game, knight, self.color):
                score += 1
        
        opp_color = "W" if self.color == "B" else "B"
        for knight in opp:
            if self.is_defending(game, knight, opp_color):
                score -= 1
        
        return score

    def mobility(self, game: Knights):
        opp_color = "W" if self.color == "B" else "B"
        return len(game.all_actions()[self.color]) - len(game.all_actions()[opp_color])

    def tempo(self, game: Knights):
        return 1 if game.turn == self.color else 0

    def material(self, game: Knights):
        mine, opp = (game.whiteKnights, game.blackKnights) if self.color == "W" else (game.blackKnights, game.whiteKnights)
        return len(mine) - len(opp)

    def threat(self, game: Knights):
        mine, opp = (game.whiteKnights, game.blackKnights) if self.color == "W" else (game.blackKnights, game.whiteKnights)
        deltas = [ (2, 1), (-2, 1), (2, -1), (-2, -1),
                   (1, 2), (-1, 2), (1, -2), (-1, -2) ]

        score = 0
        for knight in mine:
            row, col = knight // self.board_size, knight % self.board_size
            for dr, dc in deltas:
                if (row + dr) * self.board_size + col + dc in opp:
                    score += 1
        
        for knight in opp:
            row, col = knight // self.board_size, knight % self.board_size
            for dr, dc in deltas:
                if (row + dr) * self.board_size + col + dc in mine:
                    score -= 1
        
        return score