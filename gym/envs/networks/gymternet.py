import gym
import networkx as nx
import numpy as np
import collections
from gym.spaces import Box


class GymternetIntranet(gym.Env):
    """
    Environment representing a self-managed intranet
    """

    def __init__(self):
        super(GymternetIntranet, self).__init__()
        self.is_initialized = False
        self.cumul_reward = 0

    def set_topology(self, topology, low=1e-8, high=1e2):
        if self._is_wrong_type(topology) :
            raise ValueError("`topology` argument must be a list of lists")

        dim_size = len(topology[0])
        if dim_size != 2:
            raise ValueError("`topology` argument must be a list or tuple")

        self.network = nx.Graph(topology)
        self.n_edges = len(self.network.edges())
        self.n_nodes = len(self.network.nodes())

        gym.Env.action_space = Box(low, high, shape=(self.n_edges,), dtype=np.float32)
        gym.Env.observation_space = Box(low, high, shape=(self.n_nodes, self.n_nodes), dtype=np.float32)
        self.is_initialized = True

    def _check_is_initialized(self):
            if not self.is_initialized:
                raise ValueError("Must define a topology through set_topology method before using this object")


    def _is_wrong_type(self, obj):
        """
        Return true if object is not a list
        """
        return not (isinstance(obj, collections.Collection) or self.isinstance(obj[0], collections.Collection))

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (object): an action provided by the environment

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (boolean): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        self._check_is_initialized()
        demand = self._get_observation()
        reward = self._get_reward(demand, action)
        self.cumul_reward += reward
        done = False   # Check conditions in calling function
        info = {}      # Not needed
        return demand, reward, done, info

    def reset(self):
        self._check_is_initialized()
        self.cumul_reward = 0
        return self._get_observation()

    def _get_reward(self, demand, action):
        """
        TODO: replace with implementation from http://www.cs.huji.ac.il/~schapiram/Learning_to_Route%20(NIPS).pdf
        :return: Random uniform demand matrix of size self.n_nodes x self.n_nodes scaled by `scale_factor`
        """
        return 1.0


    def _get_observation(self):
        """
        :return: Random uniform demand matrix of size self.n_nodes x self.n_nodes scaled by `scale_factor`
        """
        return gym.Env.observation_space.sample()

