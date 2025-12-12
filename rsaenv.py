import numpy as np
import pandas as pd

import gymnasium as gym
from gymnasium import spaces

from typing import List, Tuple, Dict, Optional

from nwutil import Request, generate_sample_graph, LinkState


# Helper: path definitions

# We follow the problem statement's path IDs:
# P1: 0,1,2,3
# P2: 0,8,7,6,3
# P3: 0,1,5,4
# P4: 0,8,7,6,3,4
# P5: 7,1,2,3
# P6: 7,6,3
# P7: 7,1,5,4
# P8: 7,6,3,4
#
# We map each discrete action (0..7) to one of these paths.
ACTION_TO_PATH: List[Dict] = [
    {"src": 0, "dst": 3, "nodes": [0, 1, 2, 3]},        # action 0 -> P1
    {"src": 0, "dst": 3, "nodes": [0, 8, 7, 6, 3]},     # action 1 -> P2
    {"src": 0, "dst": 4, "nodes": [0, 1, 5, 4]},        # action 2 -> P3
    {"src": 0, "dst": 4, "nodes": [0, 8, 7, 6, 3, 4]},  # action 3 -> P4
    {"src": 7, "dst": 3, "nodes": [7, 1, 2, 3]},        # action 4 -> P5
    {"src": 7, "dst": 3, "nodes": [7, 6, 3]},           # action 5 -> P6
    {"src": 7, "dst": 4, "nodes": [7, 1, 5, 4]},        # action 6 -> P7
    {"src": 7, "dst": 4, "nodes": [7, 6, 3, 4]},        # action 7 -> P8
]


def _edges_from_path(nodes: List[int]) -> List[Tuple[int, int]]:
    """Convert a node sequence to a list of undirected edges with sorted endpoints."""
    edges: List[Tuple[int, int]] = []
    for u, v in zip(nodes[:-1], nodes[1:]):
        if u > v:
            u, v = v, u
        edges.append((u, v))
    return edges


class RSAEnv(gym.Env):
    """
    Custom Gymnasium environment for the Routing and Spectrum Allocation (RSA) problem.

    - Each episode corresponds to ONE request CSV file with 100 requests.
    - Each time step t processes exactly one request.
    - The agent picks a path (one of the pre-defined paths P1..P8).
    - The environment tries to assign the smallest-index wavelength that is
      available on all links along the chosen path (wavelength continuity).
    - Reward is +1 for accepted requests and 0 for blocked requests.
      (Maximizing cumulative reward == minimizing request blocking.)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        capacity: int,
        request_files: List[str],
        max_holding_time: Optional[int] = None,
        seed: int = 0,
    ) -> None:
        super().__init__()

        assert len(request_files) > 0, "At least one request file is required."
        self.link_capacity = capacity
        self.request_files = request_files
        self.rng = np.random.default_rng(seed)

        # Build topology with LinkState objects
        self.graph = generate_sample_graph(link_capacity=self.link_capacity)
        self.num_nodes = self.graph.number_of_nodes()

        # Fixed ordering of undirected edges for observation encoding
        self.edges: List[Tuple[int, int]] = []
        for u, v in self.graph.edges():
            if u > v:
                u, v = v, u
            self.edges.append((u, v))
        self.edges = sorted(self.edges)
        self.num_edges = len(self.edges)

        # Requests for current episode
        self.requests: List[Request] = []
        self.current_request: Optional[Request] = None
        self.current_step: int = 0
        self.total_steps: int = 0

        # Episode statistics
        self.blocked_count: int = 0

        # Holding time normalisation (used only for scaling)
        if max_holding_time is not None:
            self.max_holding_time = max_holding_time
        else:
            self.max_holding_time = 100  # safe upper bound

        # ---- Gym spaces ----
        # Observation:
        #   - link wavelengths occupancy: shape = (num_edges * capacity,), values in {0,1}
        #   - current request source one-hot: shape = (num_nodes,)
        #   - current request destination one-hot: shape = (num_nodes,)
        #   - normalised holding time: scalar
        obs_dim = self.num_edges * self.link_capacity + 2 * self.num_nodes + 1
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # Action: choose one of the 8 predefined paths P1..P8
        self.action_space = spaces.Discrete(len(ACTION_TO_PATH))

    # Internal helpers

    def _reset_link_states(self) -> None:
        """Reset all LinkState objects on every edge."""
        for u, v in self.edges:
            state: LinkState = self.graph.edges[u, v]["state"]
            state.reset(self.link_capacity)

    def _free_expired_wavelengths(self) -> None:
        """Free all wavelengths whose holding time has expired at current_step."""
        for u, v in self.edges:
            state: LinkState = self.graph.edges[u, v]["state"]
            state.free_expired(self.current_step)

    def _load_requests_from_csv(self, file_path: str) -> List[Request]:
        """Load one CSV file into a list of Request objects."""
        df = pd.read_csv(file_path)
        reqs: List[Request] = []
        for _, row in df.iterrows():
            reqs.append(
                Request(
                    int(row["source"]),
                    int(row["destination"]),
                    int(row["holding_time"]),
                )
            )
        return reqs

    def _sample_request_file(self) -> str:
        """Pick one training request file at random for this episode."""
        return self.rng.choice(self.request_files)

    def _current_request_onehot(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Encode current request as (src_onehot, dst_onehot, ht_norm)."""
        src_onehot = np.zeros(self.num_nodes, dtype=np.float32)
        dst_onehot = np.zeros(self.num_nodes, dtype=np.float32)
        if self.current_request is not None:
            s = self.current_request.source
            d = self.current_request.destination
            src_onehot[s] = 1.0
            dst_onehot[d] = 1.0
            ht = float(self.current_request.holding_time)
        else:
            ht = 0.0
        ht_norm = np.clip(ht / float(self.max_holding_time), 0.0, 1.0)
        return src_onehot, dst_onehot, ht_norm

    def _get_obs(self) -> np.ndarray:
        """Build a flattened observation vector."""
        # 1) Link occupancy (0/1)
        occ_list = []
        for u, v in self.edges:
            state: LinkState = self.graph.edges[u, v]["state"]
            # binary occupancy of each wavelength
            occ = (state.wavelength_occupancy != 0).astype(np.float32)
            # In case capacity in LinkState and env mismatch, pad or trim
            if occ.shape[0] < self.link_capacity:
                occ = np.pad(
                    occ,
                    (0, self.link_capacity - occ.shape[0]),
                    mode="constant",
                    constant_values=0.0,
                )
            elif occ.shape[0] > self.link_capacity:
                occ = occ[: self.link_capacity]
            occ_list.append(occ)
        link_occ = np.concatenate(occ_list, dtype=np.float32)

        # 2) Current request encoding
        src_oh, dst_oh, ht_norm = self._current_request_onehot()
        ht_arr = np.array([ht_norm], dtype=np.float32)

        return np.concatenate([link_occ, src_oh, dst_oh, ht_arr], dtype=np.float32)

    def _get_info(self) -> Dict:
        """Extra information for debugging and logging."""
        blocking_rate = (
            float(self.blocked_count) / float(self.total_steps)
            if self.total_steps > 0
            else 0.0
        )
        return {
            "step": self.current_step,
            "blocked_count": self.blocked_count,
            "blocking_rate": blocking_rate,
        }

    # Gym API

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Start a new episode with a fresh request file and empty network."""
        super().reset(seed=seed)
        if seed is not None:
            # Update RNG for reproducibility when needed
            self.rng = np.random.default_rng(seed)

        # Choose a request CSV:
        # - If options contains {"request_file": path}, use that file.
        # - Otherwise, randomly sample one from self.request_files.
        if options is not None and "request_file" in options:
            file_path = options["request_file"]
        else:
            file_path = self._sample_request_file()

        self.requests = self._load_requests_from_csv(file_path)
        self.total_steps = len(self.requests)
        self.current_step = 0
        self.blocked_count = 0

        # Reset all links
        self._reset_link_states()

        # Load the first request
        self.current_request = self.requests[0] if self.total_steps > 0 else None

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: int):
        """
        Apply the chosen action (path selection) for the current request, update network
        state, and return (observation, reward, terminated, truncated, info).
        """
        assert self.current_request is not None, "No active request in environment."

        # 1) Advance logical time and free expired lightpaths
        self._free_expired_wavelengths()

        req = self.current_request
        src, dst = req.source, req.destination

        # 2) Interpret action as one of the predefined paths
        valid_action = 0 <= action < len(ACTION_TO_PATH)
        success = False

        if valid_action:
            path_desc = ACTION_TO_PATH[action]
            if path_desc["src"] == src and path_desc["dst"] == dst:
                path_nodes = path_desc["nodes"]
                success = self._try_allocate_path(path_nodes, req.holding_time)
            else:
                # action corresponds to a path for a different OD pair -> invalid
                success = False
        else:
            success = False

        # 3) Reward and stats
        if success:
            reward = 1.0
        else:
            reward = 0.0
            self.blocked_count += 1

        # 4) Move to next request / time slot
        self.current_step += 1
        terminated = self.current_step >= self.total_steps
        truncated = False

        if terminated:
            # No more requests in this episode
            self.current_request = None
            obs = self._get_obs()
        else:
            self.current_request = self.requests[self.current_step]
            obs = self._get_obs()

        info = self._get_info()
        return obs, reward, terminated, truncated, info

    # RSA core: spectrum allocation

    def _try_allocate_path(self, path_nodes: List[int], holding_time: int) -> bool:
        """
        Try to allocate the smallest-index wavelength that is available on
        all links of the given path.

        Returns True on success (request accepted) and False if blocked.
        """
        edges = _edges_from_path(path_nodes)
        release_time = self.current_step + int(holding_time)

        for color in range(self.link_capacity):
            # Check feasibility on all links
            feasible = True
            link_states: List[LinkState] = []
            for u, v in edges:
                # Each edge is undirected and stored with sorted endpoints
                state: LinkState = self.graph.edges[u, v]["state"]
                if not state.is_color_free(color):
                    feasible = False
                    break
                link_states.append(state)

            if feasible:
                # Allocate this wavelength on every link
                for state in link_states:
                    state.allocate(color, release_time)
                return True

        # No common free wavelength -> request blocked
        return False
