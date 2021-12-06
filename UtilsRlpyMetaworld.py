__copyright__ = "Copyright 2013, RLPy http://www.acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = "Ray N. Forcement"

from rlpy.Tools import plt, mpatches, fromAtoB
from rlpy.Domains.Domain import Domain
import numpy as np
class ChainMDPTut(Domain):
    """
    Tutorial Domain - nearly identical to ChainMDP.py
    """
    #: Reward for each timestep spent in the goal region
    GOAL_REWARD = 0
    #: Reward for each timestep
    STEP_REWARD = -1
    #: Set by the domain = min(100,rows*cols)
    episodeCap = 0
    # Used for graphical normalization
    MAX_RETURN = 1
    # Used for graphical normalization
    MIN_RETURN  = 0
    # Used for graphical shifting of arrows
    SHIFT       = .3
    #:Used for graphical radius of states
    RADIUS      = .5
    # Stores the graphical pathes for states so that we can later change their colors
    circles     = None
    #: Number of states in the chain
    chainSize   = 0
    # Y values used for drawing circles
    Y           = 1