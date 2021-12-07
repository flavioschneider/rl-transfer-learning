#!/usr/bin/env python3
"""This is an example to train TRPO on ML1 Push environment."""
# pylint: disable=no-value-for-parameter
import click
import metaworld
import torch
import sys
sys.path.append('/home/ico/Desktop/RL/rl-transfer-learning/garage-master/src')


from garage import wrap_experiment
from garage.envs import normalize
from garage.envs.multi_env_wrapper import MultiEnvWrapper, round_robin_strategy
from garage.experiment.deterministic import set_seed
from garage.experiment.task_sampler import MetaWorldTaskSampler
from garage.sampler import LocalSampler
from garage.torch.algos import TRPO, HORDE
from garage.torch.policies import GaussianMLPPolicy
from garage.torch.value_functions import GaussianMLPValueFunction
from garage.trainer import Trainer
from garage.replay_buffer import PathBuffer
from garage.torch.q_functions import ContinuousMLPQFunction
from garage.np.exploration_policies import EpsilonGreedyPolicy


@click.command()
@click.option('--seed', default=1)
@click.option('--epochs', default=500)
@click.option('--batch_size', default=1024)
@wrap_experiment(snapshot_mode='all')
def horde_metaworld_mt1_push(ctxt, seed=1, epochs=500, batch_size=1024):
    """Set up environment and algorithm and run the task.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.
        epochs (int): Number of training epochs.
        batch_size (int): Number of environment steps in one batch.

    """
    set_seed(seed)
    n_tasks = 1
    mt1 = metaworld.MT1('push-v1')
    train_task_sampler = MetaWorldTaskSampler(mt1, 'train',
                                              lambda env, _: normalize(env))
    envs = [env_up() for env_up in train_task_sampler.sample(n_tasks)]
    env = MultiEnvWrapper(envs,
                          sample_strategy=round_robin_strategy,
                          mode='vanilla')

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(64, 64),
        hidden_nonlinearity=torch.tanh,
        output_nonlinearity=None,
    )

    sampler = LocalSampler(agents=policy,
                         envs=env,
                         max_episode_length=env.spec.max_episode_length)

    replay_buffer = PathBuffer(
        capacity_in_transitions=1e4)

    qf = ContinuousMLPQFunction(env_spec=env.spec,
                                hidden_sizes=(32, 32),
                                hidden_nonlinearity=torch.tanh,
                                output_nonlinearity=None
                                )
    n_epochs = epochs
    steps_per_epoch = 1
    sampler_batch_size = batch_size
    num_timesteps = n_epochs * steps_per_epoch * sampler_batch_size
    exploration_policy = EpsilonGreedyPolicy(
        env_spec=env.spec,
        policy=policy,
        total_timesteps=num_timesteps,
        max_epsilon=1.0,
        min_epsilon=0.01,
        decay_ratio=0.1)

    algo = HORDE(env_spec=env.spec,
                policy=policy,
                exploration_policy=exploration_policy,
                qf=qf,
                replay_buffer=replay_buffer,
                sampler=sampler)

    trainer = Trainer(ctxt)
    trainer.setup(algo, env)
    trainer.train(n_epochs=epochs, batch_size=batch_size)


horde_metaworld_mt1_push()
