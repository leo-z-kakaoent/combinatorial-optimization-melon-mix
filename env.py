import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

rawdf = pd.read_parquet("./data/twodays_played_small.parquet")
user_played_jnp_list = [jnp.array(np.unique(rawdf.iloc[i].content_id_s)) for i in range(len(rawdf))]

# 전체를 대표할수 있는 100개의 곡 30개 모음을 만드는게 목표
num_clusters = 256
num_songs_per_cluster = 30

# num_songs_per_cluster + 1 을 가지는 action size를 만들어서 각 곡이 어떤 클러스터에 들어갈지 결정함
class SongClustersEnv(py_environment.PyEnvironment):
    def __init__(self, user_played_jnp_list, songs, num_clusters, num_songs_per_cluster):
        self._user_played_jnp_list = user_played_jnp_list
        self._songs = songs
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=num_clusters, name='action'
        )
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(num_clusters*2+3,), dtype=np.float32, name="observation"
        )
        self._num_clusters = num_clusters
        self._num_songs_per_cluster = num_songs_per_cluster
        self._state = {
            "songs_per_cluster": [jnp.array([]) for _ in range(self._num_clusters)],
            "occurence_per_cluster": {i: None for i in range(self._num_clusters)},
        }
        self._candidate_i = 0
        self._observation = None
        self._candidate_occurence = self._occur(self._candidate_i)
        self._episode_ended = False
        
    def action_spec(self):
        return self._action_spec
    
    def observation_spec(self):
        return self._observation_spec
    
    def observation(self):
        return self._observation
    
    def _reset(self):
        self._state = {
            "songs_per_cluster": [jnp.array([]) for _ in range(self._num_clusters)],
            "occurence_per_cluster": {i: None for i in range(self._num_clusters)},
        }
        self._candidate_i = 0
        self._candidate_occurence = self._occur(self._candidate_i)
        self._observation = self._observe()
        self._episode_ended = False
        return ts.restart
    
    def _occur(self):
        candidate = self._songs[self._candidate_i]
        return jnp.array([jnp.isin(candidate, played, assume_unique=True) for played in self._user_played_jnp_list])
    
    def _observe(self):
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
        popularity = jnp.mean(self._candidate_occurence)
        rank = self._candidate_i / len(self._songs)
        taken_seats = jnp.sum([len(songs) for songs in self._state["songs_per_cluster"]]) 
        left_seats = taken_seats / self._num_clusters / self._num_songs_per_cluster
        any_co_occurences = []
        all_co_occurences = []
        for i in range(self._num_clusters):
            an, al = co_occur(i, self._candidate_occurence)
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
        mean_per_user_cluster = jnp.column_stack([jnp.mean(v, axis=1) for v in self._state['occurence_per_cluster'].values])
        max_per_user = jnp.max(mean_per_user_cluster, axis=1)
        return jnp.mean(max_per_user)
    
    def _check_episode_end(self):
        if jnp.all([len(songs) == self._num_songs_per_cluster for songs in self._state['songs_per_cluster']]):
            done = True
        else:
            done = False
        return done
    
    def _actionable_space(self):
        return jnp.array([len(songs) < self._num_songs_per_cluster for songs in self._state['songs_per_cluster']])
    
    def _update_state(self, action):
        # songs per cluster에 곡을 추가하고
        self._state['songs_per_cluster'][action] = jnp.append(self._state['songs_per_cluster'][action], self._songs[self._candidate_i])
        # occurence per cluster에 해당하는 곡의 occurence도 추가한다
        self._state['occurence_per_cluster'][action] = jnp.append(self._state['occurence_per_cluster'][action], self._candidate_occurence, axis=1)
    
    def _step(self, action):
        actionable_space = self._actionable_space()
        if action == self._num_clusters:
            action_possible = True
        else:
            if actionable_space[action]:
                action_possible = True
            else:
                action_possible = False
                
        if action_possible:
            if action == self._num_clusters: # Passing Action
                self._candidate_i += 1
                self._candidate_occurence = self._occur(self._candidate_i)
                self._observation = self._observe()
            else:
                self._update_state(action)
                self._candidate_i += 1
                self._candidate_occurence = self._occur(self._candidate_i)
                self._observation = self._observe()
                
        if self._check_episode_end():
            return ts.termination(self._observation, self._reward())
        else:
            return ts.transition(self._observation, reward=0.0, discount=1.0)