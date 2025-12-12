# import networkx as nx

# class Request:
#     def __init__(self, source: int, destination: int, holding_time: int):
#         self.source = source
#         self.destination = destination
#         self.holding_time = holding_time

#     def __repr__(self):
#         return f"Request(src={self.source}, dst={self.destination}, hold={self.holding_time})"


# class BaseLinkState:
#     def __init__(self, u, v, capacity=20, utilization=0.0):
#         if u > v: # sort by the node ID
#             u, v = v, u
#         self.endpoints = (u, v)
#         self.capacity = capacity
#         self.utilization = utilization

#     def __repr__(self):
#         return f"LinkState(capacity={self.capacity}, util={self.utilization})"

# class LinkState(BaseLinkState):
#     """ 
#     Data structure to store the link state.
#     You can extend this class to add more attributes if needed.
#     Do not change the BaseLinkState class.
#     """
#     def __init__(self, u, v, capacity=20, utilization=0.0):
#         super().__init__(u, v, capacity, utilization) 


# def generate_sample_graph():
#     # Create the sample graph
#     G = nx.Graph()

#     G.add_nodes_from(range(9))

#     # Define links: ring links + extra links
#     links = [(n, (n + 1) % 9) for n in range(9)] + [(1, 7), (1, 5), (3, 6)]

#     # Add edges with link state objects
#     for u, v in links:
#         G.add_edge(u, v, state=LinkState(u, v))
#     return G


import networkx as nx
import numpy as np


class Request:
    def __init__(self, source: int, destination: int, holding_time: int):
        self.source = source
        self.destination = destination
        self.holding_time = holding_time

    def __repr__(self):
        return f"Request(src={self.source}, dst={self.destination}, hold={self.holding_time})"


class BaseLinkState:
    def __init__(self, u, v, capacity=20, utilization=0.0):
        if u > v:  # sort by the node ID
            u, v = v, u
        self.endpoints = (u, v)
        self.capacity = capacity
        self.utilization = utilization

    def __repr__(self):
        return f"LinkState(capacity={self.capacity}, util={self.utilization})"


class LinkState(BaseLinkState):
    """
    Data structure to store the link state.

    You can extend this class to add more attributes if needed.
    Do not change the BaseLinkState class.
    """

    def __init__(self, u, v, capacity=20, utilization=0.0):
        super().__init__(u, v, capacity, utilization)
        # Each wavelength slot has:
        # - wavelength_occupancy[c] in {0,1}: 1 means occupied, 0 means free
        # - wavelength_release[c] : release time step for this slot, -1 if free
        self.wavelength_occupancy = np.zeros(self.capacity, dtype=np.int8)
        self.wavelength_release = -np.ones(self.capacity, dtype=np.int32)

    # Helpers used by the RSA environment

    def reset(self, capacity: int = None) -> None:
        """
        Reset this link state to an empty link. Optionally update capacity.
        """
        if capacity is not None and capacity != self.capacity:
            self.capacity = int(capacity)
        self.wavelength_occupancy = np.zeros(self.capacity, dtype=np.int8)
        self.wavelength_release = -np.ones(self.capacity, dtype=np.int32)
        self.utilization = 0.0

    def free_expired(self, current_time: int) -> None:
        """
        Free all wavelength slots whose release time is <= current_time.
        """
        expired = self.wavelength_release <= current_time
        if np.any(expired):
            self.wavelength_occupancy[expired] = 0
            self.wavelength_release[expired] = -1
        # Update utilization as fraction of used wavelengths
        self.utilization = float(self.wavelength_occupancy.sum()) / float(self.capacity)

    def is_color_free(self, color: int) -> bool:
        """
        Return True if the given wavelength index is free on this link.
        """
        if color < 0 or color >= self.capacity:
            return False
        return self.wavelength_occupancy[color] == 0

    def allocate(self, color: int, release_time: int) -> bool:
        """
        Allocate a given wavelength slot if available.

        Returns True on success and False if the slot was already occupied
        or the color index is out of range.
        """
        if not self.is_color_free(color):
            return False
        self.wavelength_occupancy[color] = 1
        self.wavelength_release[color] = int(release_time)
        self.utilization = float(self.wavelength_occupancy.sum()) / float(self.capacity)
        return True


def generate_sample_graph(link_capacity: int = 20):
    """
    Create the sample network topology used in the RSA project.

    Parameters
    ----------
    link_capacity : int
        Number of wavelength slots on every link in the network. This is the
        capacity parameter used in Parts 1 and 2 of the project.
    """
    # Create the sample graph
    G = nx.Graph()

    # 9 nodes: 0..8
    G.add_nodes_from(range(9))

    # Define links: ring links + extra links
    links = [(n, (n + 1) % 9) for n in range(9)] + [(1, 7), (1, 5), (3, 6)]

    # Add edges with link state objects
    for u, v in links:
        G.add_edge(u, v, state=LinkState(u, v, capacity=link_capacity))
    return G
