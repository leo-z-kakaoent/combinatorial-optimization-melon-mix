import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.trajectories import time_step as ts

rawdf = pd.read_parquet("./data/twodays_played_small.parquet")
user_played_jnp_list = [jnp.array(np.unique(rawdf.iloc[i].content_id_s)) for i in range(len(rawdf))]

def occur(user_played_jnp_list, candidate):
    occur_bool = [jnp.isin(candidate, played, assume_unique=True) for played in user_played_jnp_list]
    return jnp.array(occur_bool)

candidate = jnp.array(30089801)
occurence = occur(user_played_jnp_list, candidate)

a = jnp.array([])
b = jnp.zeros(shape=(10000,1))
jnp.append(a,jnp.zeros(shape=(10000,1)),axis=1)
jnp.append(b,b,axis=1).shape

# 전체를 대표할수 있는 100개의 곡 30개 모음을 만드는게 목표
num_clusters = 256
num_songs_per_cluster = 30

# num_songs_per_cluster + 1 을 가지는 action size를 만들어서 각 곡이 어떤 클러스터에 들어갈지 결정함
class SongClustersEnv(py_environment.PyEnvironment):
    def __init__(self, user_played_jnp_list, num_clusters, num_songs_per_cluster):
        self._user_played_jnp_list = user_played_jnp_list
        self._action_spec = None
        self._observation_spec = None
        self._num_clusters = num_clusters
        self._num_songs_per_cluster = num_songs_per_cluster
        self._state = {
            "songs_per_cluster": [jnp.array([]) for _ in range(self._num_clusters)],
            "occurence_per_cluster": {i: None for i in range(self._num_clusters)},
        }
        self._candidate_i = 0
        self._episode_ended = False
        
    def action_spec(self):
        return self._action_spec
    
    def observation_spec(self):
        return self._observation_spec
    
    def _reset(self):
        self._state = {
            "songs_per_cluster": [jnp.array([]) for _ in range(self._num_clusters)],
            "occurence_per_cluster": {i: None for i in range(self._num_clusters)},
        }
        self._candidate_i = 0
        self._episode_ended = False
    
    # 새로 추가할 여부를 판단해야하는 candiate를 받아서,
    # candidate가 각 유저에게 얼마나 재생되었는지 데이터는 이미 있고. (init 에서 계산해놓음)
    # 각 클러스터의 엘레먼트가 candidate와 co-occurence가 어떻게 되는지 카운트해야함
    def _observe_candidate(self, candidate):
        for c in self._state:
            if len(c) > 0:
                temp = jnp.concatenate([candidate,c])
                jnp.isin(temp, )
    
    def _occur(self, candidate):
        return jnp.array([jnp.isin(candidate, played, assume_unique=True) for played in self._user_played_jnp_list])
    
    # reset:
    # 1. Popularity 기준으로 순서대로 song id를 가져옴
    # 2. Observation 만들기
    #  2.1. Song를 기준으로 Occurence Array를 만듬
    #  2.2. Occurence Array의 Sum으로 Popularity 가져오기 (1)
    #  2.2. 각 Cluster의 co-occurence 를 계산해서 sum함 (100) Any 기준
    #  2.3. All 기준 (100)
    
    # agent가 action를 뱉어줌
    
    # step:
    # 1. action을 받아서 state를 업데이트함
    # 2. 업데이트된 state를 기준으로 done 체크
    #  2.1. done이면 리워드 계산
    # 3. song_id를 업데이트함
    # 3. 