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
    def __init__(self, user_played_jnp_list, songs, num_clusters, num_songs_per_cluster):
        self._user_played_jnp_list = user_played_jnp_list
        self._songs = songs
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
    
    def _occur(self, candidate):
        return jnp.array([jnp.isin(candidate, played, assume_unique=True) for played in self._user_played_jnp_list])
    
    def _observe(self, candidate):
        # 2. Observation 만들기
        #  2.1. Song를 기준으로 Occurence Array를 만듬
        #  2.2. Occurence Array의 Sum으로 Popularity 가져오기 (1) popularity
        #  2.3. candidate_i 를 len(num_songs) 에 나눔 (1) rank
        #  2.4. 남은 자리수를 num_cluster로 나눔 (1) left_seats
        #  2.2. 각 Cluster의 co-occurence 를 계산해서 sum함 (100) Any 기준 any_co_occurences
        #  2.3. All 기준 (100) all_co_occurences
        def co_occur(i, candidate_occurence):
            cluster_occurences = self._state['occurence_per_cluster'][i]
            if cluster_occurences is not None:
                co_occurences = cluster_occurences[candidate_occurence]
                any_co_occurence = jnp.mean(jnp.any(co_occurences, axis=1))
                all_co_occurence = jnp.mean(jnp.all(co_occurences, axis=1))
            else:
                any_co_occurence = jnp.zeros(1)
                all_co_occurence = jnp.zeros(1)
            return any_co_occurence, all_co_occurence
        candidate_occurence = self._occur(candidate)
        popularity = jnp.mean(candidate_occurence)
        rank = self._candidate_i / len(self._songs)
        taken_seats = jnp.sum([len(songs) for songs in self._state["songs_per_cluster"]]) 
        left_seats = taken_seats / self._num_clusters / self._num_songs_per_cluster
        any_co_occurences = []
        all_co_occurences = []
        for i in range(self._num_clusters):
            an, al = co_occur(i, candidate_occurence)
            any_co_occurences.append(an)
            all_co_occurences.append(al)
        
        observations = []
        observations.append(popularity)
        observations.append(rank)
        observations.append(left_seats)
        observations += any_co_occurences
        observations += all_co_occurences
        return jnp.array(observations)
    
    def _reward(self):    
        # reward 계산식
        # 각 유저별로 각 클러스터가 커버되는 비율을 계산하고 그 맥스만 모음
        #  - Occurence_per_cluster의 행의 mean을 구하고 stack함
        #  - 맥스를 구하고 다시 mean함
        mean_per_user_cluster = jnp.column_stack([jnp.mean(v, axis=1) for v in self._state['occuence_per_cluster'].values])
        max_per_user = jnp.max(mean_per_user_cluster, axis=1)
        return jnp.mean(max_per_user)
    
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
    # 4. 업데이트된 song_id 기준으로 observation 만듬
    # 5. observation, 리워드 리턴하기
    
    # done 조건
    # 1. 모든 클러스터에 곡이 찼을때 (리워드 계산)
    # 2. 곡 리스트가 다 소모되었을때 (리워드 0)
    #  2.1. 남은 곡과 채워져야하는 곡수를 계산해서 넣기? (굳이 코딩할 필요 있나?)
    
    # reward 계산식
    # 각 유저별로 각 클러스터가 커버되는 비율을 계산하고 그 맥스만 모음
    #  - Occurence_per_cluster의 행의 mean을 구하고 stack함
    #  - 맥스를 구하고 다시 mean함

a = []   
a += [1]