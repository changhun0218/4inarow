# -*- coding: utf-8 -*-
"""
Monte Carlo Tree Search in AlphaGo Zero style, which uses a policy-value
network to guide the tree search and evaluate the leaf nodes

@author: Junxiao Song
"""

import numpy as np
import copy
import time

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
                ##print("Winner is :", self.whose_turn)
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

    def __init__(self, policy_value_fn, c_puct=5, n_playout=1000):
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
        v = np.random.rand()
        p = np.random.dirichlet([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
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

    def __init__(self, policy_value_function=None,
                 c_puct=5, n_playout=2000, is_selfplay=0):
        self.mcts = MCTS(policy_value_function, c_puct, n_playout)
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
            probs = probs * array_available
            probs /= np.sum(probs)
            move_probs[list(acts)] = probs # p(a)
            
            self.res_board.append(board.get_actual_board()) # store boards for learning
            self.res_probs.append(move_probs) # store probability vectors for learning
            if self._is_selfplay:
                # add Dirichlet Noise for exploration (needed for
                # self-play training)
                move = np.random.choice(
                    acts,
                    p=0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs)))
                )
                # update the root node and reuse the search tree
                self.mcts.update_with_move(move, board)
            else:
                # with the default temp=1e-3, it is almost equivalent
                # to choosing the move with the highest prob
                move = np.random.choice(acts, p=probs)
                # reset the root node
                self.mcts.update_with_move(-1)
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

if __name__=="__main__":
    start_time = time.time()
    play = MCTSPlayer(is_selfplay=True)
    board = Board()
    input_ = np.zeros((6,7,)) # input for DNN
    output_ = np.array([]) # answer for DNN
    z_temp = 1
    for i in range(43):
        move = play.get_action(board)
        end, winner = board.game_end()
        pi = np.append(play.res_probs[-1], z_temp) # pi, v: pi which is prob of a
        output_ = np.append(output_,pi)
        # print("who's turn:",z_temp)
        if end or (i==42 ):
            #print("winner:", board.game_end()[1])
            #winner_arr = np.ones((len(play.res_board))) * winner
            #print(winner_arr)
            print("who wins?:", z_temp)
            break

        s = np.array(play.res_board[-1]) # state
        input_ = np.c_[input_, s] 
        #print(s)
        #print(pi)
        z_temp *= -1

    input_ = input_.reshape(6,-1,7)
    output_ = output_.reshape(-1,8)
    if z_temp == -1:    # if -1 wins, revert the 'v'
        output_[:,7] *= -1
    else:
        pass
    
    #print("input:",input_.shape)
    print("output:",output_)
 
    np.save("input",input_)  # input[:,# of move,:]
    np.save("output", output_) # output[# of move, :]
    end_time = time.time()    
    print(end_time - start_time)
