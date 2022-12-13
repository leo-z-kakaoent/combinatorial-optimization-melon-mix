import sys
sys.path.insert(1, '/workspace/combinatorial-optimization-melon-mix')
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from collections import Counter
from environment import SongClustersEnv


rawdf = pd.read_parquet("./data/twodays_played_small.parquet")
user_played_jnp_list = [np.unique(rawdf.iloc[i].content_id_s) for i in range(len(rawdf))]

# 전체를 대표할수 있는 100개의 곡 30개 모음을 만드는게 목표
num_clusters = 5
num_songs_per_cluster = 5

count_dict = Counter(np.concatenate(rawdf.content_id_s.values))
songs = [v[0] for v in count_dict.most_common()]

env = SongClustersEnv(user_played_jnp_list, songs, num_clusters, num_songs_per_cluster)
env.reset()

env._state
env.step(1)
env.step(2)

import time
c = 0
going = True
while going:
    r = env.step(np.random.randint(env._num_clusters+1))
    if (c % 100 == 0) or (r.reward > 0):
        print([len(it) for it in env._state['songs_per_cluster']])
        print()
        print(r.reward)
        print()
        time.sleep(1)
    c += 1
    
    
import reverb
import tensorflow as tf
from tf_agents.agents.ppo import ppo_actor_network
from tf_agents.agents.ppo import ppo_clip_agent
from tf_agents.replay_buffers import reverb_utils
from tf_agents.networks import value_network
from tf_agents.train.utils import spec_utils
from tf_agents.train.utils import train_utils

#class ReverbFixedLengthSequenceObserver(reverb_utils.ReverbAddTrajectoryObserver):

actor_fc_layers = (64,64)
value_fc_layers = (64,64)
importance_ratio_clipping = 0.2
lambda_value = 0.95
discount_factor = 0.99
entropy_regularization = 0.
value_pred_loss_coef = 0.5
num_epoch = 1
use_gae = True
use_td_lambda_return = True
gradient_clipping = 0.5
value_clipping = None

collect_env = SongClustersEnv(user_played_jnp_list, songs, num_clusters, num_songs_per_cluster)
eval_env = SongClustersEnv(user_played_jnp_list, songs, num_clusters, num_songs_per_cluster)

observation_tensor_spec, action_tensor_spec, time_step_tensor_spec = spec_utils.get_tensor_specs(collect_env)

train_step = train_utils.create_train_step()

actor_net_builder = ppo_actor_network.PPOActorNetwork()
actor_net = actor_net_builder.create_sequential_actor_net(actor_fc_layers, action_tensor_spec)
value_net = value_network.ValueNetwork(
    observation_tensor_spec,
    fc_layer_params=value_fc_layers,
    kernel_initializer=tf.keras.initializers.Orthogonal()
)

current_iteration = tf.Variable(0, dtype=tf.int64)
def learning_rate_fn():
    return learning_rate * (1-current_iteration / num_iterations)

agent = ppo_clip_agent.PPOClipAgent(
    time_step_tensor_spec,
    action_tensor_spec,
    optimizer=tf.optimizers.Adam(learning_rate=learning_rate_fn, epsilon=1e-5),
    actor_net=actor_net,
    value_net=value_net,
    importance_ratio_clipping=importance_ratio_clipping,
    lambda_value=lambda_value,
    discount_factor=discount_factor,
    entropy_regularization=entropy_regularization,
    value_pred_loss_coef=value_pred_loss_coef,
    num_epochs=num_epoch,
    use_gae=use_gae,
    use_td_lambda_return=use_td_lambda_return,
    gradient_clipping=gradient_clipping,
    value_clipping=value_clipping,
    compute_value_and_advantage_in_train=False,
    update_normalizers_in_train=False,
    debug_summaries=False,
    summarize_grads_and_vars=False,
    train_step_counter=train_step,
)
agent.initialize()
