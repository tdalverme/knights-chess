from Knights import Knights
import random, numpy, time

'''
Things to do:
Things done:
1 - Zobrist keys and hashing
2 - Score computing
3 - Quiscence search
4 - Alpha - Beta pruning
5 - History
6 - Killer moves
7 - Capture moves
8 - Move ordering
9 - Iterative deepening
'''

class BrianPlayer:
    # Constants utilized in the class
    WIN = 1000
    LOSE = -1000
    AMOUNT_PIECES = 32
    KILLER_MOVES = 3
    MAX_DEPTH = 5000
    HASH_TABLE_MAXLEN = 2^30 - 1
    VALUE_POSITIONS =   [
                        1, 1, 1, 1, 1, 1, 1, 1,
                        2, 2, 2, 2, 2, 2, 2, 2,
                        3, 3, 3, 3, 3, 3, 3, 3,
                        4, 6, 7, 7, 7, 7, 6, 4,
                        4, 6, 7, 7, 7, 7, 6, 4,
                        3, 3, 3, 3, 3, 3, 3, 3,
                        2, 2, 2, 2, 2, 2, 2, 2,
                        1, 1, 1, 1, 1, 1, 1, 1
                        ]

    def __init__(self, color, board_size):
        self.color = color
        self.board_size = board_size
        self.time_per_turn = 3

        # If we want to print the amount of nodes visited
        self.counter = 0

        # Used in zobrist hashing
        self.zobrist_matrix = numpy.zeros((64, 64, 2))
        self.init_table()
        self.hash_table = {}

        # History_table[number_o[f_players][kfrom][kto]
        self.history_table = numpy.zeros((2, (self.board_size*self.board_size), (self.board_size*self.board_size)), dtype=int)

        # Used in killer moves
        self.killer_moves = numpy.empty((self.MAX_DEPTH, self.KILLER_MOVES), dtype=object)
        self.init_killer_moves()

        # Used in capturing moves
        self.capture_moves = []

        self.zob_is_black_turn  = random.randint(0, 1)

    def init_table(self):
        # We init the zobrist_matrix with randoms int for each cell
        for row in range(self.board_size):
            for col in range(self.board_size):
                for k in range(2):
                    self.zobrist_matrix[row][col][k] = random.randint(0, 2**64 - 1)
        
    def init_killer_moves(self):
        # We init the killer_moves matrix with (0,0) tuples
        for row in range(self.MAX_DEPTH):
            for col in range(self.KILLER_MOVES):
                self.killer_moves[row][col] = (0,0)

    def compute_hash(self, game):
        hash = 0

        # For each white piece in the board, we get the value of the cell and compute a hash
        for knight in game.whiteKnights:
            row, col = knight // self.board_size, knight % self.board_size
            cell_value = self.zobrist_matrix[row][col][0]
            hash ^= numpy.int64(cell_value)
        
        # For each black piece in the board, we get the value of the cell and compute a hash
        for knight in game.blackKnights:
            row, col = knight // self.board_size, knight % self.board_size
            cell_value = self.zobrist_matrix[row][col][1]
            hash ^= numpy.int64(cell_value)
        
        if game.turn == "B":
            hash ^= self.zob_is_black_turn
        
        return hash

    def score(self, game):

        if game.game_over() != "-":
            if game.game_over() == self.color:
                return self.WIN
            else:
                return self.LOSE

        white_positioning_total = 0
        black_positioning_total = 0

        for knight in game.whiteKnights:
            white_positioning_total += self.VALUE_POSITIONS[knight]
        
        for knight in game.blackKnights:
            black_positioning_total += self.VALUE_POSITIONS[knight]

        if game.turn == "W":
            black_positioning_total *= -1
        else:
            white_positioning_total *= -1

        # If both teams have the same amount of knights, then no points for extra_pieces
        # Otherwise, we compute that difference
        extra_pieces = abs(len(game.whiteKnights) - len(game.blackKnights))

        capture_score = 0

        capturing_moves = self.generate_moves(game)
        for kfrom, kto in capturing_moves:
            own_pieces, op_pieces = self.check_defenders(game, kfrom)
            own_pieces_2, op_pieces_2 = self.check_defenders(game, kto)
            capture_score = own_pieces + own_pieces_2 - op_pieces - op_pieces_2
        
        final_score = (white_positioning_total + black_positioning_total + capture_score * extra_pieces)

        # If we are maxing then return the plain score, otherwise we return the opposite
        if game.turn == self.color:
            return final_score
        else:
            return -final_score

    def check_defenders(self, game, position):
        # This function calcutes the amount of pieces defending a certain position
        deltas = [ (2, 1), (-2, 1), (2, -1), (-2, -1),
                   (1, 2), (-1, 2), (1, -2), (-1, -2) ]
        
        self_pieces = 0
        enemy_pieces = 0

        knights, op_knights = (game.whiteKnights, game.blackKnights) if game.turn == "W" else (game.blackKnights, game.whiteKnights)

        for dr, dc in deltas:
            row, col = position // self.board_size, position % self.board_size
            if 0 <= row+dr and row+dr < self.board_size and 0 <= col+dc and col+dc < self.board_size:
                kdefender = (row+dr)*8 + col+dc
                if kdefender in knights:
                    self_pieces += 1
                elif kdefender in op_knights:
                    enemy_pieces += 1

        return (self_pieces, enemy_pieces)

    def sort_actions(self, moves, game):
        turn_to_play = 0 if game.turn == "W" else 1

        # To generate a list with capture moves
        self.capture_moves = self.generate_moves(game)

        for row in range(len(self.history_table[turn_to_play])):
            for col in range(len(self.history_table[turn_to_play])):
                value = self.history_table[turn_to_play][row][col]
                if value != 0 and (row, col) in moves:
                    moves.remove((row, col))
                    moves.insert(0, (row, col))

        # We prioritize killer moves before history moves
        for array in self.killer_moves:
            first_move = array[0]
            second_move = array[1]
            third_move = array[2]

            if third_move in moves:
                moves.remove(third_move)
                moves.insert(0, third_move)
            if second_move in moves:
                moves.remove(second_move)
                moves.insert(0, second_move)
            if first_move in moves:
                moves.remove(first_move)
                moves.insert(0, first_move)
        
        for move in self.capture_moves:
            if move in moves:
                moves.remove(move)
                moves.insert(0, move)

        
        hash = self.compute_hash(game)
        if hash in self.hash_table.keys():
            move = self.hash_table[hash]["best_move"]
            if move in moves:
                moves.remove(move)
                moves.insert(0, move)

        return moves
    
    def isCapture(self, game: Knights, move):
        knights, op_knights = (game.whiteKnights, game.blackKnights) if game.turn == "W" else (game.blackKnights, game.whiteKnights)

        kfrom, kto = move

        # If there is an opponent piece in kto, then it's a capture and we return True
        if kto in op_knights:
            return True
        
        return False

    def get_move(self, game: Knights):
        best_value= -10000
        self.current_depth = 0
        start_time = time.time()
        max_time = start_time + self.time_per_turn
        best_move = None

        # We compute the hash value of the position to store it
        hash = self.compute_hash(game)

        # We try to find the best action possible for the depth
        while time.time() < max_time:
            for move in self.sort_actions(game.all_actions()[game.turn], game):
                game.make_action(move)

                local_value = -self.minmax_enhanced(game, self.current_depth, 10000, -10000)

                # If the value of the current position is better than the value of the best position
                # we found a better position and we have to keep the value and the move
                if local_value > best_value:
                    best_value = local_value
                    best_move = move

                game.undo_action(move)

                if time.time() >= max_time and self.current_depth > 1:
                    self.insert_in_hashTable(hash, best_value, move, self.current_depth)
                    print("Depth: " + str(self.current_depth))
                    return best_move
            
            self.current_depth += 1

        # We store the position in the hash table
        self.insert_in_hashTable(hash, best_value, move, self.current_depth)

        print("Depth: " + str(self.current_depth))

        # We return the best move
        return best_move

    def minmax_enhanced(self, game, depth, alpha, beta):
        # To print the amount of nodes visited
        global counter
        self.counter += 1

        # Base case
        if game.game_over() != "-":
            if game.game_over() == self.color:
                return self.WIN
            else:
                return self.LOSE
        
        # Compute the hash number of this instant of the game
        hash = self.compute_hash(game)

        # If depth is reached, then we do a quiscence search and return the best value found
        if depth == 0:
            value = -self.quis(game, 5, -beta, -alpha)
            self.insert_in_hashTable(hash, value, None, depth)
            return self.hash_table[hash]["value"]
        
        # We check if the hash is already in the table, meaning that this position was already considered so
        # we return it's value
        # If the depth of the hash move is lower than the actual depth, we have no use for it. We continue with the minmax
        if hash in self.hash_table and depth <= self.hash_table[hash]["depth"]:
            value = self.hash_table[hash]["value"]
            return value
        
        # If it's the white's turn then we order the actions for importance, make the action, call the recursive
        # function and then we undo. We try to obtain the best alpha possible and we add the position 
        # to the hash table
        for move in self.sort_actions(game.all_actions()[game.turn], game):
            game.make_action(move)
            value = -self.minmax_enhanced(game, depth - 1, -beta, -alpha)
            game.undo_action(move)
            if value > alpha:
                alpha = value
            if alpha >= beta:
                if not self.isCapture(game, move):
                    self.update_historyArray(game, move, depth)
                    self.insert_killerMove(move, depth)
                    break
        self.insert_in_hashTable(hash, alpha, move, depth)
        return alpha
    
    def quis(self, game, depth, alpha, beta):
        # We evaluate the position
        best_val = self.score(game)

        if depth == 0:
            return best_val

        # If the evaluation indicates a value higher than beta, we beta cut the sub-tree and we update the 
        # history array
        if best_val >= beta:
            return best_val
        
        # If the evaluation indicates a value better than alpha, we replace it
        if best_val > alpha:
            alpha = best_val

        # Finding only the moves that captures another piece
        capturing_moves = self.generate_moves(game)

        # For each capture move, we make the action, call the recursive function and we undo
        for move in capturing_moves:
            game.make_action(move)
            eval_plus = self.score(game)
            game.undo_action(move)
            if eval_plus > alpha:
                act_val = -self.quis(game, depth - 1, -beta, -alpha)
                if act_val > best_val:
                    best_val = act_val
                    if best_val >= beta:
                        return best_val
                    if best_val > alpha:
                        alpha = best_val
            elif eval_plus > best_val:
                best_val = eval_plus

        return best_val

    # Function to find the moves that captures another piece for both whites and blacks
    def generate_moves(self, game):
        capturing_moves = []

        for move in game.all_actions()[game.turn]:

            if self.isCapture(game, move):
                capturing_moves.append(move)

        return capturing_moves

    def update_historyArray(self, game, move, depth):
        kfrom, kto = move

        # 0 for whites and 1 for blacks
        turn_to_play = 0 if game.turn == "W" else 1

        self.history_table[turn_to_play][kfrom][kto] += depth*depth
    
    def insert_killerMove(self, move, depth):
        self.killer_moves[depth][2] = self.killer_moves[depth][1]
        self.killer_moves[depth][1] = self.killer_moves[depth][0]
        self.killer_moves[depth][0] = move

    def insert_in_hashTable(self, hash, value, move, depth):

        index = hash % self.HASH_TABLE_MAXLEN

        self.hash_table[hash] = {"index": index, "value": value, "best_move": move, "depth": depth}
