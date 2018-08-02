import numpy as np
import copy
import time
import tensorflow as tf

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

class Board():
    """ Board of a four in row
    This makes a move on a board and check if it ends"""
    
    whose_turn = -1
    vectors = np.array([[1, 0], [1, 1], [0, 1], [1, -1]])

    def __init__(self):
        self.board = np.zeros((7, 9,))
        self.last_move = [None, None]
        self.winner = None
        self.availables = np.arange(7)

    def board_setting(self):
        pass
    
    def board_update(self, i):
        num_non_zero = np.count_nonzero(self.board[:, i + 1])
        x, y = 5 - num_non_zero, i + 1
        self.last_move = np.array([x, y])
        if num_non_zero == 6: # over
            return

        self.board[x, y] = self.whose_turn
        self.availables = np.where(self.board[0][1:8] == 0)  # indicators which shows the available move. 

    def game_end(self):
        pos = self.last_move
        if pos[0] == None:
            return False, False

        if len(self.availables) == 0:
            self.winner = 0
            return True, self.winner

        for vector in self.vectors:
            pos = self.last_move
            count = 0
            for sign in [-1, 1]:
                temp = np.copy(pos)
                while count < 4:
                    temp += vector * sign
                    if self.board[temp[0], temp[1]] == self.whose_turn:
                        count += 1
                    else :
                        break
            if count >= 3:
                #print(self.board)
                #print("available:", self.availables)
                self.winner = self.whose_turn
                #print("Winner is :", self.whose_turn)
                return True, self.winner
        return False, None
    
    def make_a_move(self, action):
        self.whose_turn *= -1
        self.board_update(action)

    def get_board(self):
        return self.board

    def get_actual_board(self):
        return self.board[:-1, 1:-1]
    
    def get_current_player(self):
        return self.whose_turn
    
    def get_move(self):
        pass

    
class TreeNode(object):
    """A node in the MCTS tree.

    Each node keeps track of its own value Q, prior probability P, and
    its visit-count-adjusted prior score u.
    """

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

    def expand(self, action_priors):
        """Expand tree by creating new children.
        action_priors: a list of tuples of actions and their prior probability
            according to the policy function.
        """
        action_priors = zip(np.arange(7), action_priors)
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        """Select action among children that gives maximum action value Q
        plus bonus u(P).
        Return: A tuple of (action, next_node)
        """
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        """Update node values from leaf evaluation.
        leaf_value: the value of subtree evaluation from the current player's
            perspective.
        """
        # Count visit.
        self._n_visits += 1
        # Update Q, a running average of values for all visits.
        self._Q += 1.0*(leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        """Like a call to update(), but applied recursively for all ancestors.
        """
        # If it is not root, this node's parent should be updated first.
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        """Calculate and return the value for this node.
        It is a combination of leaf evaluations Q, and this node's prior
        adjusted for its visit count, u.
        c_puct: a number in (0, inf) controlling the relative impact of
            value Q, and prior probability P, on this node's score.
        """
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded)."""
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS(object):
    """An implementation of Monte Carlo Tree Search."""

    def __init__(self, sess, c_puct=5, n_playout=500):
        """
        policy_value_fn: a function that takes in a board state and outputs
            a list of (action, probability) tuples and also a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        """
        self._root = TreeNode(None, 1.0)
        #self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _policy(self, state):
        pred = sess.run(tf_y_pred, feed_dict = {tf_x: state.get_actual_board().reshape(1, -1)}).reshape(-1)
        p = softmax(pred[:7])
        v = pred[7]
#        v = np.random.rand()
#        p = np.random.dirichlet([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        return p, v
                    
        
    def _playout(self, state):
        """Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """
        node = self._root
        while(1):
            if node.is_leaf():
                break
            # Greedily select next move.
            action, node = node.select(self._c_puct)
            state.make_a_move(action)
        end, winner = state.game_end()
            
        # Evaluate the leaf using a network which outputs a list of
        # (action, probability) tuples p and also a score v in [-1, 1]
        # for the current player.
        action_probs, leaf_value = self._policy(state)
        if not end:
            node.expand(action_probs)

        else:
            # for end state，return the "true" leaf_value
            if winner == -1:  # tie
                leaf_value = 0.0
            else:
                leaf_value = (
                    1.0 if winner == state.get_current_player() else -1.0
                )

        # Update value and visit count of nodes in this traversal.
        node.update_recursive(-leaf_value)

    def get_move_probs(self, state, temp=1e-3):
        """Run all playouts sequentially and return the available actions and
        their corresponding probabilities.
        state: the current game state
        temp: temperature parameter in (0, 1] controls the level of exploration
        """
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)

        # calc the move probabilities based on visit counts at the root node
        act_visits = [(act, node._n_visits)
                      for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))

        return acts, act_probs

    def update_with_move(self, last_move, board):
        """
        Step forward in the tree, keeping everything we already know
        about the subtree.
        """
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
            board.make_a_move(last_move)
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"


    

class MCTSPlayer(object):
    """AI player based on MCTS"""

    def __init__(self, sess, c_puct=5, n_playout=1000, is_selfplay=0):
        self.mcts = MCTS(sess, c_puct, n_playout)
        self._is_selfplay = is_selfplay
        self.res_board = []
        self.res_probs = []

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board, temp=1.0, return_prob=0):
        sensible_moves = board.availables
        #sensible_moves = np.arange(7)
        # the pi vector returned by MCTS as in the alphaGo Zero paper
        move_probs = np.zeros((7))
        if len(sensible_moves) > 0:
            acts, probs = self.mcts.get_move_probs(board, temp) # a, p_a
            # to choose only available moves
            array_available = np.zeros((7))
            array_available[sensible_moves] = 1
#            probs = probs * array_available
#            probs /= np.sum(probs)
            move_probs[list(acts)] = probs # p(a)
            
            self.res_board.append(board.get_actual_board()) # store boards for learning
            self.res_probs.append(move_probs) # store probability vectors for learning
            if self._is_selfplay:
                # add Dirichlet Noise for exploration (needed for
                # self-play training)
                explore_prob = 0.75 * probs + 0.25 * np.random.dirichlet(0.3 * np.ones(len(probs)))
                explore_prob = explore_prob * array_available
                explore_prob /= np.sum(explore_prob)
                print(explore_prob)
                move = np.random.choice(
                    acts,
                    p = explore_prob
                )
                # update the root node and reuse the search tree
                self.mcts.update_with_move(move, board)
                print(board.get_actual_board())
            else:
                # with the default temp=1e-3, it is almost equivalent
                # to choosing the move with the highest prob
                probs = probs * array_available
                probs /= np.sum(probs)
                move = np.random.choice(acts, p=probs)
                # reset the root node
                self.mcts.update_with_move(move, board)
                #                location = board.move_to_location(move)
                #                print("AI move: %d,%d\n" % (location[0], location[1]))
            if return_prob:
                return move, move_probs
            else:
                return move
        else:
            return None
            print("WARNING: the board is full")

    def __str__(self):
        return "MCTS {}".format(self.player)

def game_game(sess):
    board = Board()
    play = MCTSPlayer(sess, is_selfplay=False)
    z_temp = 1
    
    for i in range(43):
        my_move = int(input("Your turn:"))
        board.make_a_move(my_move)
        print(board.get_actual_board())
        end, winner = board.game_end()

        if end or (i==42):
            #print("winner:", board.game_end()[1])
            #winner_arr = np.ones((len(play.res_board))) * winner
            #print(winner_arr)
            print("who wins?:", z_temp)
            break
        
        move = play.get_action(board)
        end, winner = board.game_end()
        print(board.get_actual_board())
        
        # print("who's turn:",z_temp)

def cnn_layer(x, kernel):
    w_conv = tf.Variable(tf.truncated_normal(kernel, stddev=0.1))
    conv = tf.nn.conv2d(input = x,
                        filter = w_conv,
                        strides = [1, 1, 1, 1],
                        padding = "SAME"
    )
    b_conv = tf.Variable(tf.constant(0.1, shape=[kernel[3]]))

    conv_rl = tf.nn.relu(conv + b_conv)
    return conv_rl


if __name__=="__main__":
    tf_x = tf.placeholder(tf.float32, [None, 42])
    tf_y = tf.placeholder(tf.float32, [None, 8])
    tf_pi, tf_z = tf.split(tf_y, [7,1], 1)
    tf_image = tf.reshape(tf_x, [-1,6,7,1])
    
    array_kernel = [[3, 3, 1, 32], [3, 3, 32, 64], [3, 3, 64, 1024]]
    conv = tf_image
    for kernel in array_kernel:
        conv = cnn_layer(conv, kernel)

    #fully connected
    result = tf.reshape(conv,[-1,6*7*1024])
    tf_y_pred = tf.contrib.layers.fully_connected(result, 8)
    tf_p, tf_v = tf.split(tf_y_pred, [7,1], 1)

    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, "./tmp/model.ckpt")

    while(1):
        game_game(sess)
