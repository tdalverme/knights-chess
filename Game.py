from Knights import Knights
from TomasPlayer import Mode, TomasPlayer
from TomasPlayer2 import Mode2, TomasPlayer2
from BrianPlayer import BrianPlayer

if __name__ == "__main__":
      game = Knights(8)
      player1 = BrianPlayer("W", 8)
      # player2 = BrianPlayer("B", 8)
      # player1 = TomasPlayer(game, "W", 8, Mode.Agressive, 4)
      player2 = TomasPlayer(game, "B", 8, Mode.Agressive, 4)
   
      print(game)
      while game.game_over() == "-":
          move = player1.get_move(game)
          game.make_action(move)

          print(game)
          if game.game_over() == "-":
             move = player2.get_move(game)
             game.make_action(move)
             print(game)