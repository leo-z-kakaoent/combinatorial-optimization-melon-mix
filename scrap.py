import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment

jnp.array([])

rawdf = pd.read_parquet("./data/twodays_played_small.parquet")

np.unique(np.concatenate(rawdf.content_id_s.values)).shape

# 전체를 대표할수 있는 100개의 곡 30개 모음을 만드는게 목표
num_clusters = 256
num_songs_per_cluster = 30

# num_songs_per_cluster + 1 을 가지는 action size를 만들어서 각 곡이 어떤 클러스터에 들어갈지 결정함
class SongClustersEnv(py_environment.PyEnvironment):
    def __init__(self, num_clusters, num_songs_per_cluster):
        self._action_spec = None
        self._observation_spec = None
        self._num_clusters = num_clusters
        self._num_songs_per_cluster = num_songs_per_cluster
        self._state = [jnp.array([]) for _ in range(self._num_clusters)]
        self._episode_ended = False
        
    def action_spec(self):
        return self._action_spec
    
    def observation_spec(self):
        return self._observation_spec
    
    def _reset(self):
        self._state = [jnp.array([]) for _ in range(self._num_clusters)]
        self._episode_ended = False
        
    def _
    