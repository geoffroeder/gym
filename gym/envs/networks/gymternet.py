import gym
import networkx as nx
import numpy as np
import collections


class GymternetIntranet(gym.Env):
    """
    Environment representing a self-managed intranet
    """

    def __init__(self, topology):
        if self._is_wrong_type(topology) :
            raise ValueError("`topology` argument must be a list or tuple of lists or tuples")

        dim_size = len(topology[0])
        if dim_size != 2:
            raise ValueError("`topology` argument must be a list or tuple")

        self.network = nx.G(topology)
        self.n_edges = len(self.network.edges())
        self.n_nodes = len(self.network.nodes())

    def _is_wrong_type(self, obj):
        """
        Return true if object is not a list
        """
        return isinstance(obj, collections.List) or self.isinstance(obj[0], collections.Sequence)

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
        demand = self._get_random_observation()
        reward = self._get_reward(demand, action)
        done = True
        return demand, reward, done, None

    def _get_reward(self, demand, action):
        """
        TODO: replace with implementation from http://www.cs.huji.ac.il/~schapiram/Learning_to_Route%20(NIPS).pdf
        :return: Random uniform demand matrix of size self.n_nodes x self.n_nodes scaled by `scale_factor`
        """
        return 1.0


    def _get_random_observation(self, scale_factor=1e3):
        """
        TODO: replace with implementation from http://www.cs.huji.ac.il/~schapiram/Learning_to_Route%20(NIPS).pdf
        :return: Random uniform demand matrix of size self.n_nodes x self.n_nodes scaled by `scale_factor`
        """
        return np.random.rand(self.n_nodes, self.n_nodes) * scale_factor

