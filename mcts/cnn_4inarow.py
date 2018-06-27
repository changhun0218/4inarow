import numpy as np
import tensorflow as tf
from mcts_4inarow import *

if __name__ == "__main__":
    play = MCTSPlayer(is_selfplay=True)
    board = Board()
    n_games = 1 # number of games for reinforcement learning
    for _ in range(n_games):
        while True: # execute a single game and collect results
            move = play.get_action(board)
            print(play.res_board[-1])
            print(play.res_probs[-1])
            print("winner:", board.game_end()[1])
            end, winner = board.game_end()
            if end:
                winner_arr = np.ones((len(play.res_board))) * winner
                print(winner_arr)
                break
        # this is just an example
        tf_x = tf.placeholder(tf.float32, [None, 6 * 7]) 
        image = tf.reshape(tf_x, [-1, 6, 7, 1])  # (batch, height, width, channel)
        tf_y = tf.placeholder(tf.int32, [None, 2]
