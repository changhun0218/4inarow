"""
Monte Carlo Tree Search (MCTS)
"""

"""
MCTS will be executed for a single move and store the information of the move.
The information of moves will be used for neural network to learn
MCTS also will be guided by the neural network.
"""

import numpy as np

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
        self._u = (c_puct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u
    
    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded)."""
        return self._children == {}

    def is_root(self):
        return self._parent is None
    

def DoMove(s, a): # move from position s

def GetMove(s): # returns possible moves
    
        
def f(s):
    v = np.random.rand()
    p = np.random.dirichlet([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    return p, v
        
class MCTS:
    def __init__(self, s):
        self.s = s
        self.root_s = s
        # initialization
        """
        N(s, a) : a visit count
        W(s, a) : a total action value
        P(s, a) : a prior probability
        Q(s, a) : a mean action value
        """
    def N(self, s, a):
        
        
    def select(self):
        
        """
        a move for Monte Carlo simulation will be determined that maximizes Q + U
        return 0~6
        """

    def expand(self):
        a = np.where(np.random.multinomial(1, f(self.s)[0]) == 1)
        self.s = DoMove(self.s, a)
        node = Node(s, a)
        
    def eval(self):
        
    def backup(self):
        
