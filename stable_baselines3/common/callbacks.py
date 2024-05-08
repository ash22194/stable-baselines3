import os
import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union
import sys
import gymnasium as gym
import numpy as np
import torch as th

from stable_baselines3.common.logger import Logger

try:
	from tqdm import TqdmExperimentalWarning

	# Remove experimental warning
	warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
	from tqdm.rich import tqdm
except ImportError:
	# Rich not installed, we only throw an error
	# if the progress bar is used
	tqdm = None

from stable_baselines3.common import base_class
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization


class BaseCallback(ABC):
	"""
	Base class for callback.

	:param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
	"""

	# The RL model
	# Type hint as string to avoid circular import
	model: "base_class.BaseAlgorithm"

	def __init__(self, verbose: int = 0):
		super().__init__()
		# Number of time the callback was called
		self.n_calls = 0  # type: int
		# n_envs * n times env.step() was called
		self.num_timesteps = 0  # type: int
		self.verbose = verbose
		self.locals: Dict[str, Any] = {}
		self.globals: Dict[str, Any] = {}
		# Sometimes, for event callback, it is useful
		# to have access to the parent object
		self.parent = None  # type: Optional[BaseCallback]

	@property
	def training_env(self) -> VecEnv:
		training_env = self.model.get_env()
		assert (
			training_env is not None
		), "`model.get_env()` returned None, you must initialize the model with an environment to use callbacks"
		return training_env

	@property
	def logger(self) -> Logger:
		return self.model.logger

	# Type hint as string to avoid circular import
	def init_callback(self, model: "base_class.BaseAlgorithm") -> None:
		"""
		Initialize the callback by saving references to the
		RL model and the training environment for convenience.
		"""
		self.model = model
		self._init_callback()

	def _init_callback(self) -> None:
		pass

	def on_training_start(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
		# Those are reference and will be updated automatically
		self.locals = locals_
		self.globals = globals_
		# Update num_timesteps in case training was done before
		self.num_timesteps = self.model.num_timesteps
		self._on_training_start()

	def _on_training_start(self) -> None:
		pass

	def on_rollout_start(self) -> None:
		self._on_rollout_start()

	def _on_rollout_start(self) -> None:
		pass

	def _before_step(self) -> None:
		# the locals() call happens after the step, so assume no access to local variables here
		pass

	@abstractmethod
	def _on_step(self) -> bool:
		"""
		:return: If the callback returns False, training is aborted early.
		"""
		return True

	def _after_step(self) -> None:
		pass

	def before_step(self) -> None:
		"""
		This method will be called before stepping the environment
		no updated local variables available, can be used for timing calls
		"""
		self._before_step()

	def on_step(self) -> bool:
		"""
		This method will be called by the model after each call to ``env.step()``.

		For child callback (of an ``EventCallback``), this will be called
		when the event is triggered.

		:return: If the callback returns False, training is aborted early.
		"""
		self.n_calls += 1
		self.num_timesteps = self.model.num_timesteps

		return self._on_step()

	def after_step(self) -> None:
		"""
		This method will be called at the end of the rollout loop
		no updated local variables available, can be used for timing calls
		"""
		self._after_step()

	def on_training_end(self) -> None:
		self._on_training_end()

	def _on_training_end(self) -> None:
		pass

	def on_rollout_end(self) -> None:
		self._on_rollout_end()

	def _on_rollout_end(self) -> None:
		pass

	def update_locals(self, locals_: Dict[str, Any]) -> None:
		"""
		Update the references to the local variables.

		:param locals_: the local variables during rollout collection
		"""
		self.locals.update(locals_)
		self.update_child_locals(locals_)

	def update_child_locals(self, locals_: Dict[str, Any]) -> None:
		"""
		Update the references to the local variables on sub callbacks.

		:param locals_: the local variables during rollout collection
		"""
		pass


class EventCallback(BaseCallback):
	"""
	Base class for triggering callback on event.

	:param callback: Callback that will be called
		when an event is triggered.
	:param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
	"""

	def __init__(self, callback: Optional[BaseCallback] = None, verbose: int = 0):
		super().__init__(verbose=verbose)
		self.callback = callback
		# Give access to the parent
		if callback is not None:
			assert self.callback is not None
			self.callback.parent = self

	def init_callback(self, model: "base_class.BaseAlgorithm") -> None:
		super().init_callback(model)
		if self.callback is not None:
			self.callback.init_callback(self.model)

	def _on_training_start(self) -> None:
		if self.callback is not None:
			self.callback.on_training_start(self.locals, self.globals)

	def _on_event(self) -> bool:
		if self.callback is not None:
			return self.callback.on_step()
		return True

	def _on_step(self) -> bool:
		return True

	def update_child_locals(self, locals_: Dict[str, Any]) -> None:
		"""
		Update the references to the local variables.

		:param locals_: the local variables during rollout collection
		"""
		if self.callback is not None:
			self.callback.update_locals(locals_)


class CallbackList(BaseCallback):
	"""
	Class for chaining callbacks.

	:param callbacks: A list of callbacks that will be called
		sequentially.
	"""

	def __init__(self, callbacks: List[BaseCallback]):
		super().__init__()
		assert isinstance(callbacks, list)
		self.callbacks = callbacks

	def _init_callback(self) -> None:
		for callback in self.callbacks:
			callback.init_callback(self.model)

	def _on_training_start(self) -> None:
		for callback in self.callbacks:
			callback.on_training_start(self.locals, self.globals)

	def _on_rollout_start(self) -> None:
		for callback in self.callbacks:
			callback.on_rollout_start()

	def _on_step(self) -> bool:
		continue_training = True
		for callback in self.callbacks:
			# Return False (stop training) if at least one callback returns False
			continue_training = callback.on_step() and continue_training
		return continue_training

	def _on_rollout_end(self) -> None:
		for callback in self.callbacks:
			callback.on_rollout_end()

	def _on_training_end(self) -> None:
		for callback in self.callbacks:
			callback.on_training_end()

	def update_child_locals(self, locals_: Dict[str, Any]) -> None:
		"""
		Update the references to the local variables.

		:param locals_: the local variables during rollout collection
		"""
		for callback in self.callbacks:
			callback.update_locals(locals_)


class CheckpointCallback(BaseCallback):
	"""
	Callback for saving a model every ``save_freq`` calls
	to ``env.step()``.
	By default, it only saves model checkpoints,
	you need to pass ``save_replay_buffer=True``,
	and ``save_vecnormalize=True`` to also save replay buffer checkpoints
	and normalization statistics checkpoints.

	.. warning::

	  When using multiple environments, each call to  ``env.step()``
	  will effectively correspond to ``n_envs`` steps.
	  To account for that, you can use ``save_freq = max(save_freq // n_envs, 1)``

	:param save_freq: Save checkpoints every ``save_freq`` call of the callback.
	:param save_path: Path to the folder where the model will be saved.
	:param name_prefix: Common prefix to the saved models
	:param save_replay_buffer: Save the model replay buffer
	:param save_vecnormalize: Save the ``VecNormalize`` statistics
	:param verbose: Verbosity level: 0 for no output, 2 for indicating when saving model checkpoint
	"""

	def __init__(
		self,
		save_freq: int,
		save_path: str,
		name_prefix: str = "rl_model",
		save_replay_buffer: bool = False,
		save_vecnormalize: bool = False,
		verbose: int = 0,
	):
		super().__init__(verbose)
		self.save_freq = save_freq
		self.save_path = save_path
		self.name_prefix = name_prefix
		self.save_replay_buffer = save_replay_buffer
		self.save_vecnormalize = save_vecnormalize

	def _init_callback(self) -> None:
		# Create folder if needed
		if self.save_path is not None:
			os.makedirs(self.save_path, exist_ok=True)

	def _checkpoint_path(self, checkpoint_type: str = "", extension: str = "") -> str:
		"""
		Helper to get checkpoint path for each type of checkpoint.

		:param checkpoint_type: empty for the model, "replay_buffer_"
			or "vecnormalize_" for the other checkpoints.
		:param extension: Checkpoint file extension (zip for model, pkl for others)
		:return: Path to the checkpoint
		"""
		return os.path.join(self.save_path, f"{self.name_prefix}_{checkpoint_type}{self.num_timesteps}_steps.{extension}")

	def _on_step(self) -> bool:
		if self.n_calls % self.save_freq == 0:
			model_path = self._checkpoint_path(extension="zip")
			self.model.save(model_path)
			if self.verbose >= 2:
				print(f"Saving model checkpoint to {model_path}")

			if self.save_replay_buffer and hasattr(self.model, "replay_buffer") and self.model.replay_buffer is not None:
				# If model has a replay buffer, save it too
				replay_buffer_path = self._checkpoint_path("replay_buffer_", extension="pkl")
				self.model.save_replay_buffer(replay_buffer_path)  # type: ignore[attr-defined]
				if self.verbose > 1:
					print(f"Saving model replay buffer checkpoint to {replay_buffer_path}")

			if self.save_vecnormalize and self.model.get_vec_normalize_env() is not None:
				# Save the VecNormalize statistics
				vec_normalize_path = self._checkpoint_path("vecnormalize_", extension="pkl")
				self.model.get_vec_normalize_env().save(vec_normalize_path)  # type: ignore[union-attr]
				if self.verbose >= 2:
					print(f"Saving model VecNormalize to {vec_normalize_path}")

		return True


class ConvertCallback(BaseCallback):
	"""
	Convert functional callback (old-style) to object.

	:param callback:
	:param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
	"""

	def __init__(self, callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], bool]], verbose: int = 0):
		super().__init__(verbose)
		self.callback = callback

	def _on_step(self) -> bool:
		if self.callback is not None:
			return self.callback(self.locals, self.globals)
		return True


class EvalCallback(EventCallback):
	"""
	Callback for evaluating an agent.

	.. warning::

	  When using multiple environments, each call to  ``env.step()``
	  will effectively correspond to ``n_envs`` steps.
	  To account for that, you can use ``eval_freq = max(eval_freq // n_envs, 1)``

	:param eval_env: The environment used for initialization
	:param callback_on_new_best: Callback to trigger
		when there is a new best model according to the ``mean_reward``
	:param callback_after_eval: Callback to trigger after every evaluation
	:param n_eval_episodes: The number of episodes to test the agent
	:param eval_freq: Evaluate the agent every ``eval_freq`` call of the callback.
	:param log_path: Path to a folder where the evaluations (``evaluations.npz``)
		will be saved. It will be updated at each evaluation.
	:param best_model_save_path: Path to a folder where the best model
		according to performance on the eval env will be saved.
	:param deterministic: Whether the evaluation should
		use a stochastic or deterministic actions.
	:param render: Whether to render or not the environment during evaluation
	:param verbose: Verbosity level: 0 for no output, 1 for indicating information about evaluation results
	:param warn: Passed to ``evaluate_policy`` (warns if ``eval_env`` has not been
		wrapped with a Monitor wrapper)
	"""

	def __init__(
		self,
		eval_env: Union[gym.Env, VecEnv],
		callback_on_new_best: Optional[BaseCallback] = None,
		callback_after_eval: Optional[BaseCallback] = None,
		n_eval_episodes: int = 5,
		eval_freq: int = 10000,
		log_path: Optional[str] = None,
		best_model_save_path: Optional[str] = None,
		deterministic: bool = True,
		render: bool = False,
		verbose: int = 1,
		warn: bool = True,
	):
		super().__init__(callback_after_eval, verbose=verbose)

		self.callback_on_new_best = callback_on_new_best
		if self.callback_on_new_best is not None:
			# Give access to the parent
			self.callback_on_new_best.parent = self

		self.n_eval_episodes = n_eval_episodes
		self.eval_freq = eval_freq
		self.best_mean_reward = -np.inf
		self.last_mean_reward = -np.inf
		self.deterministic = deterministic
		self.render = render
		self.warn = warn

		# Convert to VecEnv for consistency
		if not isinstance(eval_env, VecEnv):
			eval_env = DummyVecEnv([lambda: eval_env])  # type: ignore[list-item, return-value]

		self.eval_env = eval_env
		self.best_model_save_path = best_model_save_path
		# Logs will be written in ``evaluations.npz``
		if log_path is not None:
			log_path = os.path.join(log_path, "evaluations")
		self.log_path = log_path
		self.evaluations_results: List[List[float]] = []
		self.evaluations_timesteps: List[int] = []
		self.evaluations_length: List[List[int]] = []
		# For computing success rate
		self._is_success_buffer: List[bool] = []
		self.evaluations_successes: List[List[bool]] = []

	def _init_callback(self) -> None:
		# Does not work in some corner cases, where the wrapper is not the same
		if not isinstance(self.training_env, type(self.eval_env)):
			warnings.warn("Training and eval env are not of the same type" f"{self.training_env} != {self.eval_env}")

		# Create folders if needed
		if self.best_model_save_path is not None:
			os.makedirs(self.best_model_save_path, exist_ok=True)
		if self.log_path is not None:
			os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

		# Init callback called on new best model
		if self.callback_on_new_best is not None:
			self.callback_on_new_best.init_callback(self.model)

	def _log_success_callback(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
		"""
		Callback passed to the  ``evaluate_policy`` function
		in order to log the success rate (when applicable),
		for instance when using HER.

		:param locals_:
		:param globals_:
		"""
		info = locals_["info"]

		if locals_["done"]:
			maybe_is_success = info.get("is_success")
			if maybe_is_success is not None:
				self._is_success_buffer.append(maybe_is_success)

	def _on_step(self) -> bool:
		continue_training = True

		if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
			# Sync training and eval env if there is VecNormalize
			if self.model.get_vec_normalize_env() is not None:
				try:
					sync_envs_normalization(self.training_env, self.eval_env)
				except AttributeError as e:
					raise AssertionError(
						"Training and eval env are not wrapped the same way, "
						"see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
						"and warning above."
					) from e

			# Reset success rate buffer
			self._is_success_buffer = []

			episode_rewards, episode_lengths = evaluate_policy(
				self.model,
				self.eval_env,
				n_eval_episodes=self.n_eval_episodes,
				render=self.render,
				deterministic=self.deterministic,
				return_episode_rewards=True,
				warn=self.warn,
				callback=self._log_success_callback,
			)

			if self.log_path is not None:
				assert isinstance(episode_rewards, list)
				assert isinstance(episode_lengths, list)
				self.evaluations_timesteps.append(self.num_timesteps)
				self.evaluations_results.append(episode_rewards)
				self.evaluations_length.append(episode_lengths)

				kwargs = {}
				# Save success log if present
				if len(self._is_success_buffer) > 0:
					self.evaluations_successes.append(self._is_success_buffer)
					kwargs = dict(successes=self.evaluations_successes)

				np.savez(
					self.log_path,
					timesteps=self.evaluations_timesteps,
					results=self.evaluations_results,
					ep_lengths=self.evaluations_length,
					**kwargs,
				)

			mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
			mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
			self.last_mean_reward = float(mean_reward)

			if self.verbose >= 1:
				print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
				print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
			# Add to current Logger
			self.logger.record("eval/mean_reward", float(mean_reward))
			self.logger.record("eval/mean_ep_length", mean_ep_length)

			if len(self._is_success_buffer) > 0:
				success_rate = np.mean(self._is_success_buffer)
				if self.verbose >= 1:
					print(f"Success rate: {100 * success_rate:.2f}%")
				self.logger.record("eval/success_rate", success_rate)

			# Dump log so the evaluation results are printed with the correct timestep
			self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
			self.logger.dump(self.num_timesteps)

			if mean_reward > self.best_mean_reward:
				if self.verbose >= 1:
					print("New best mean reward!")
				if self.best_model_save_path is not None:
					self.model.save(os.path.join(self.best_model_save_path, "best_model"))
				self.best_mean_reward = float(mean_reward)
				# Trigger callback on new best model, if needed
				if self.callback_on_new_best is not None:
					continue_training = self.callback_on_new_best.on_step()

			# Trigger callback after every evaluation, if needed
			if self.callback is not None:
				continue_training = continue_training and self._on_event()

		return continue_training

	def update_child_locals(self, locals_: Dict[str, Any]) -> None:
		"""
		Update the references to the local variables.

		:param locals_: the local variables during rollout collection
		"""
		if self.callback:
			self.callback.update_locals(locals_)


class CustomEvalCallback(EventCallback):

	def __init__(
		self,
		eval_env: Union[gym.Env, VecEnv],
		callback_on_new_best: Optional[BaseCallback] = None,
		callback_after_eval: Optional[BaseCallback] = None,
		n_eval_episodes: int = 5,
		eval_freq: int = 10000,
		eval_offline: bool = True,
		log_path: Optional[str] = None,
		best_model_save_path: Optional[str] = None,
		deterministic: bool = True,
		render: bool = False,
		verbose: int = 1,
		warn: bool = True,
		fixed_starts = True,
		save_model = False,
		cleanup = False,
		time_run = None
	):
		super().__init__(callback_after_eval, verbose=verbose)

		self.callback_on_new_best = callback_on_new_best
		if self.callback_on_new_best is not None:
			# Give access to the parent
			self.callback_on_new_best.parent = self

		self.n_eval_episodes = n_eval_episodes
		self.eval_freq = eval_freq
		self.eval_offline = eval_offline
		self.best_mean_reward = -np.inf
		self.last_mean_reward = -np.inf
		self.deterministic = deterministic
		self.render = render
		self.warn = warn
		self.save_model = save_model
		self.cleanup = cleanup
		self.curr_check_point_id = 0
		self.time_run = time_run
		if (time_run=='cuda'):
			# in milli-seconds
			self.time_rollout = 0.
			self.time_step = 0.
			self.time_buffer = 0.
			self.time_eval = 0.
			self.time_train_loop = 0.

			# define events
			self.rollout_start = th.cuda.Event(enable_timing=True)
			self.step_start = th.cuda.Event(enable_timing=True)
			self.buffer_start = th.cuda.Event(enable_timing=True)
			self.eval_start = th.cuda.Event(enable_timing=True)
			self.train_loop_start = th.cuda.Event(enable_timing=True)
			self.train_start = th.cuda.Event(enable_timing=True)

			self.rollout_end = th.cuda.Event(enable_timing=True)
			self.step_end = th.cuda.Event(enable_timing=True)
			self.buffer_end = th.cuda.Event(enable_timing=True)
			self.eval_end = th.cuda.Event(enable_timing=True)
			self.train_loop_end = th.cuda.Event(enable_timing=True)
			self.train_end = th.cuda.Event(enable_timing=True)

		elif (time_run is not None):
			NotImplementedError

		self.eval_env = eval_env
		self.best_model_save_path = best_model_save_path
		try:
			if (self.eval_env.spec.id in gym.registry.keys()):
				is_gpu_env = False
			else:
				raise NotImplementedError('environment not found in gym registry')
		except AttributeError:
			is_gpu_env = True
		if (type(fixed_starts)==bool) and (fixed_starts):
			self.fixed_starts = []
			if (is_gpu_env):
				for n in range(n_eval_episodes):
					self.eval_env.reset()
					self.fixed_starts += [self.eval_env.state.clone().detach()]
				fixed_starts = np.array([ff.cpu().numpy() for ff in self.fixed_starts])
			else:
				for n in range(n_eval_episodes):
					self.eval_env.reset()
					self.fixed_starts += [self.eval_env.state.copy()]
				fixed_starts = np.array(self.fixed_starts)
			np.save(os.path.join(log_path, 'test_starts.npy'), fixed_starts, allow_pickle=False)
		elif (type(fixed_starts)==str) and (os.path.isfile(fixed_starts)):
			assert fixed_starts.endswith('.npy'), 'If supplying a file to load starts from, it must be .npy'
			self.fixed_starts = np.load(fixed_starts, allow_pickle=False, mmap_mode=None)
			if (is_gpu_env):
				self.fixed_starts = [th.as_tensor(ff, dtype=self.eval_env.th_dtype, device=self.eval_env.device) for _, ff in enumerate(self.fixed_starts)]
		else:
			self.fixed_starts = [None for ii in range(n_eval_episodes)]

		# Logs will be written in ``evaluations.npz``
		if log_path is not None:
			log_path = os.path.join(log_path, "evaluations")
		self.log_path = log_path
		self.evaluations_results: List[List[float]] = []
		self.evaluations_timesteps: List[int] = []
		self.evaluations_length: List[List[int]] = []
		# For computing success rate
		self._is_success_buffer: List[bool] = []
		self.evaluations_successes: List[List[bool]] = []
		
	def _init_callback(self) -> None:
		# Does not work in some corner cases, where the wrapper is not the same
		if not isinstance(self.training_env, type(self.eval_env)):
			warnings.warn("Training and eval env are not of the same type" f"{self.training_env} != {self.eval_env}")

		# Create folders if needed
		if self.best_model_save_path is not None:
			os.makedirs(self.best_model_save_path, exist_ok=True)
		if self.log_path is not None:
			os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

		# Init callback called on new best model
		if self.callback_on_new_best is not None:
			self.callback_on_new_best.init_callback(self.model)

	def _log_success_callback(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
		"""
		Callback passed to the  ``evaluate_policy`` function
		in order to log the success rate (when applicable),
		for instance when using HER.

		:param locals_:
		:param globals_:
		"""
		info = locals_["info"]

		if locals_["done"]:
			maybe_is_success = info.get("is_success")
			if maybe_is_success is not None:
				self._is_success_buffer.append(maybe_is_success)

	def _on_training_start(self) -> None:
		# for timing measurements only
		if (self.time_run=='cuda'):
			th.cuda.synchronize()
			self.train_start.record()

			th.cuda.synchronize()
			self.train_loop_start.record()

	def _on_rollout_start(self) -> None:
		# for timing measurements only
		if (self.time_run=='cuda'):
			th.cuda.synchronize()
			self.train_loop_end.record()
			th.cuda.synchronize()
			self.time_train_loop += self.train_loop_start.elapsed_time(self.train_loop_end)
			self.logger.record("time/train_loop", self.time_train_loop / 1000.)

			th.cuda.synchronize()
			self.rollout_start.record()

	def _before_step(self) -> None:
		# for timing measurements only
		# the locals call happens after the step, so assume no access to local variables
		if (self.time_run=='cuda'):
			th.cuda.synchronize()
			self.step_start.record()

	def _on_step(self) -> None:
		if (self.time_run=='cuda'):
			th.cuda.synchronize()
			self.step_end.record()
			th.cuda.synchronize()
			self.time_step += self.step_start.elapsed_time(self.step_end)

		continue_training = True

		# log terminal episodic info
		terminal_infos = [info_i for ii, info_i in enumerate(self.locals['infos']) if self.locals['dones'][ii]]
		if (len(terminal_infos) > 0):
			terminal_statnames = [ts for ts in terminal_infos[0].keys() if (ts.startswith('ep_'))]
			for ts in terminal_statnames:
				ts_stat = [terminal_infos[ii][ts] for ii in range(len(terminal_infos)) if (not np.isnan(terminal_infos[ii][ts]))]
				self.logger.record("rollout/" + ts, np.mean(ts_stat))
				if (ts == "ep_reward"):
					self.episode_rewards = ts_stat
				elif (ts == "ep_length"):
					self.episode_lengths = ts_stat
				elif (ts == "ep_terminal_goal_dist"):
					self.episode_final_goal_dist = ts_stat

		if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
			# Sync training and eval env if there is VecNormalize
			if self.model.get_vec_normalize_env() is not None:
				try:
					sync_envs_normalization(self.training_env, self.eval_env)
				except AttributeError as e:
					raise AssertionError(
						"Training and eval env are not wrapped the same way, "
						"see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
						"and warning above."
					) from e

			# Reset success rate buffer
			self._is_success_buffer = []
			if (self.eval_offline):
				try:
					if (self.eval_env.spec.id in gym.registry.keys()):
						episode_rewards, episode_discounted_rewards, episode_lengths, episode_final_goal_dist = self._evaluate_policy()
				except AttributeError:
					# if not a gym env then a gpu env
					if (self.time_run=='cuda'):
						th.cuda.synchronize()
						self.eval_start.record()

					episode_rewards, episode_discounted_rewards, episode_lengths, episode_final_goal_dist = self._evaluate_policy_customgpuenv()

					if (self.time_run=='cuda'):
						th.cuda.synchronize()
						self.eval_end.record()
						th.cuda.synchronize()
						self.time_eval += self.eval_start.elapsed_time(self.eval_end)
			else:
				episode_rewards = self.episode_rewards if hasattr(self, 'episode_rewards') else [-np.inf]
				episode_lengths = self.episode_lengths if hasattr(self, 'episode_lengths') else [-np.inf]
				episode_final_goal_dist = self.episode_final_goal_dist if hasattr(self, 'episode_final_goal_dist') else [-np.inf]

			if self.log_path is not None:
				assert isinstance(episode_rewards, list)
				assert isinstance(episode_lengths, list)
				self.evaluations_timesteps.append(self.num_timesteps)
				self.evaluations_results.append(episode_rewards)
				self.evaluations_length.append(episode_lengths)

				kwargs = {}
				# Save success log if present
				if len(self._is_success_buffer) > 0:
					self.evaluations_successes.append(self._is_success_buffer)
					kwargs = dict(successes=self.evaluations_successes)

				np.savez(
					self.log_path,
					timesteps=self.evaluations_timesteps,
					results=self.evaluations_results,
					ep_lengths=self.evaluations_length,
					**kwargs,
				)

				# save model
				if (self.save_model):
					check_point_id = divmod(self.model.num_timesteps+1, self.eval_freq)[0]
					if (check_point_id > self.curr_check_point_id):
						self.curr_check_point_id = check_point_id
						save_id = check_point_id*self.eval_freq
						self.model.save(os.path.join(self.log_path, 'model_' + str(save_id)))

			mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
			mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
			mean_final_goal_dist, std_final_goal_dist = np.mean(episode_final_goal_dist), np.std(episode_final_goal_dist)
			self.last_mean_reward = float(mean_reward)

			if self.verbose >= 1:
				print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
				print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
			# Add to current Logger
			self.logger.record("eval/mean_reward", float(mean_reward))
			self.logger.record("eval/mean_ep_length", mean_ep_length)
			self.logger.record("eval/mean_final_goal_dist", mean_final_goal_dist)

			if len(self._is_success_buffer) > 0:
				success_rate = np.mean(self._is_success_buffer)
				if self.verbose >= 1:
					print(f"Success rate: {100 * success_rate:.2f}%")
				self.logger.record("eval/success_rate", success_rate)

			# Dump log so the evaluation results are printed with the correct timestep
			self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
			self.logger.dump(self.num_timesteps)

			if mean_reward > self.best_mean_reward:
				self.best_timestep = self.num_timesteps
				if self.verbose >= 1:
					print("New best mean reward!")
				if self.best_model_save_path is not None:
					self.model.save(os.path.join(self.best_model_save_path, "best_model"))
				self.best_mean_reward = float(mean_reward)
				# Trigger callback on new best model, if needed
				if self.callback_on_new_best is not None:
					continue_training = self.callback_on_new_best.on_step()

			# Trigger callback after every evaluation, if needed
			if self.callback is not None:
				continue_training = continue_training and self._on_event()

		if (self.time_run=='cuda'):
			th.cuda.synchronize()
			self.buffer_start.record()

		return continue_training
	
	def _after_step(self) -> None:
		# for timing measurements only
		if (self.time_run=='cuda'):
			th.cuda.synchronize()
			self.buffer_end.record()
			th.cuda.synchronize()
			self.time_buffer += self.buffer_start.elapsed_time(self.buffer_end)

	def _on_rollout_end(self) -> None:
		# for timing measurements only
		if (self.time_run=='cuda'):
			th.cuda.synchronize()
			self.buffer_start.record() # hacky way to measure buffer advantage computation time
			th.cuda.synchronize()
			self.time_buffer += self.buffer_end.elapsed_time(self.buffer_start)

			th.cuda.synchronize()
			self.rollout_end.record()
			th.cuda.synchronize()
			self.time_rollout += self.rollout_start.elapsed_time(self.rollout_end)

			th.cuda.synchronize()
			self.train_loop_start.record()

			self.logger.record("time/step", self.time_step / 1000.)
			self.logger.record("time/eval", self.time_eval / 1000.)
			self.logger.record("time/buffer_update", self.time_buffer / 1000.)
			self.logger.record("time/rollout", self.time_rollout / 1000.)

	def _on_training_end(self) -> None:
		# for timing measurements only
		if (self.time_run=='cuda'):
			th.cuda.synchronize()
			self.train_end.record()
			th.cuda.synchronize()
			self.logger.record("time/total", self.train_start.elapsed_time(self.train_end) / 1000.)

	def _evaluate_policy_customgpuenv(self):

		episode_rewards = []
		episode_discounted_rewards = []
		episode_lengths = []
		episode_final_goal_dist = []

		for ep in range(self.n_eval_episodes):
			obs, _ = self.eval_env.reset(state=self.fixed_starts[ep])
			done = th.zeros(obs.shape[0], dtype=th.bool, device=self.eval_env.device)
			discount = th.ones(obs.shape[0], device=self.eval_env.device)
			ep_reward = th.zeros(obs.shape[0], device=self.eval_env.device)
			ep_discounted_reward = th.zeros(obs.shape[0], device=self.eval_env.device)
			ep_length = th.zeros(obs.shape[0], device=self.eval_env.device)

			while (not th.all(done)):
				# not using model.predict() here because it typecasts actions to cpu and numpy by default!
				with th.no_grad():
					action, _, _ = self.model.policy.forward(obs, deterministic=self.deterministic)
				# clip action for consistency with bounds
				action = th.clip(action, self.eval_env.th_action_space_low, self.eval_env.th_action_space_high)
				obs, reward, done_, _, info = self.eval_env.step(action)
				not_done = th.logical_not(done)
				done = done_
				reward = reward * not_done # add terminal value estimate -> if expecting episodes to be of different lengths?

				ep_reward += reward
				ep_discounted_reward += (discount*reward)
				ep_length += not_done
				discount *= self.model.gamma

			episode_rewards.append(ep_reward.clone().cpu().numpy())
			episode_discounted_rewards.append(ep_discounted_reward.clone().cpu().numpy())
			episode_lengths.append(ep_length.clone().cpu().numpy())
			if (hasattr(self.eval_env, 'get_goal_dist')):
				episode_final_goal_dist.append(self.eval_env.get_goal_dist().cpu().numpy())
			else:
				episode_final_goal_dist.append(np.zeros(obs.shape[0]))

		return np.concatenate(episode_rewards).tolist(), np.concatenate(episode_discounted_rewards).tolist(), np.concatenate(episode_lengths).tolist(), np.concatenate(episode_final_goal_dist).tolist()

	def _evaluate_policy(self):

		episode_rewards = []
		episode_discounted_rewards = []
		episode_lengths = []
		episode_final_goal_dist = []

		for ep in range(self.n_eval_episodes):
			obs, _ = self.eval_env.reset(state=self.fixed_starts[ep])
			done = False
			discount = 1
			ep_reward = 0
			ep_discounted_reward = 0
			ep_length = 0
			while (not done):
				action, _state = self.model.predict(obs, deterministic=self.deterministic)
				obs, reward, done, _, info = self.eval_env.step(action)

				self._log_success_callback(locals(), globals())

				ep_reward += reward
				ep_discounted_reward += (discount*reward)
				ep_length += 1
				discount *= self.model.gamma

			episode_rewards.append(ep_reward)
			episode_discounted_rewards.append(ep_discounted_reward)
			episode_lengths.append(ep_length)
			episode_final_goal_dist.append(0)
			if (hasattr(self.eval_env, 'get_goal_dist')):
				episode_final_goal_dist[-1] = self.eval_env.get_goal_dist()

		return episode_rewards, episode_discounted_rewards, episode_lengths, episode_final_goal_dist
	
	def _cleanup(self):
		if (self.cleanup):
			# delete models saved after the best found model
			for ff in os.listdir(self.log_path):
				if ('model' in ff):
					save_id = int(os.path.splitext(ff)[0].split('_')[-1])
					if (save_id > self.best_timestep):
						os.remove(os.path.join(self.log_path, ff))
			# set the model parameters to best performing model saved
			best_model_filepath = os.path.join(self.log_path, 'model_' + str(self.best_timestep) + '.zip')
			assert os.path.isfile(best_model_filepath), 'cannot find the best model savefile'
			self.model.set_parameters(best_model_filepath)
	
	def update_child_locals(self, locals_: Dict[str, Any]) -> None:
		"""
		Update the references to the local variables.

		:param locals_: the local variables during rollout collection
		"""
		if self.callback:
			self.callback.update_locals(locals_)


class StopTrainingOnRewardThreshold(BaseCallback):
	"""
	Stop the training once a threshold in episodic reward
	has been reached (i.e. when the model is good enough).

	It must be used with the ``EvalCallback``.

	:param reward_threshold:  Minimum expected reward per episode
		to stop training.
	:param verbose: Verbosity level: 0 for no output, 1 for indicating when training ended because episodic reward
		threshold reached
	"""

	parent: EvalCallback

	def __init__(self, reward_threshold: float, verbose: int = 0):
		super().__init__(verbose=verbose)
		self.reward_threshold = reward_threshold

	def _on_step(self) -> bool:
		assert self.parent is not None, "``StopTrainingOnMinimumReward`` callback must be used with an ``EvalCallback``"
		# Convert np.bool_ to bool, otherwise callback() is False won't work
		continue_training = bool(self.parent.best_mean_reward < self.reward_threshold)
		if self.verbose >= 1 and not continue_training:
			print(
				f"Stopping training because the mean reward {self.parent.best_mean_reward:.2f} "
				f" is above the threshold {self.reward_threshold}"
			)
		return continue_training


class EveryNTimesteps(EventCallback):
	"""
	Trigger a callback every ``n_steps`` timesteps

	:param n_steps: Number of timesteps between two trigger.
	:param callback: Callback that will be called
		when the event is triggered.
	"""

	def __init__(self, n_steps: int, callback: BaseCallback):
		super().__init__(callback)
		self.n_steps = n_steps
		self.last_time_trigger = 0

	def _on_step(self) -> bool:
		if (self.num_timesteps - self.last_time_trigger) >= self.n_steps:
			self.last_time_trigger = self.num_timesteps
			return self._on_event()
		return True


class StopTrainingOnMaxEpisodes(BaseCallback):
	"""
	Stop the training once a maximum number of episodes are played.

	For multiple environments presumes that, the desired behavior is that the agent trains on each env for ``max_episodes``
	and in total for ``max_episodes * n_envs`` episodes.

	:param max_episodes: Maximum number of episodes to stop training.
	:param verbose: Verbosity level: 0 for no output, 1 for indicating information about when training ended by
		reaching ``max_episodes``
	"""

	def __init__(self, max_episodes: int, verbose: int = 0):
		super().__init__(verbose=verbose)
		self.max_episodes = max_episodes
		self._total_max_episodes = max_episodes
		self.n_episodes = 0

	def _init_callback(self) -> None:
		# At start set total max according to number of envirnments
		self._total_max_episodes = self.max_episodes * self.training_env.num_envs

	def _on_step(self) -> bool:
		# Check that the `dones` local variable is defined
		assert "dones" in self.locals, "`dones` variable is not defined, please check your code next to `callback.on_step()`"
		self.n_episodes += np.sum(self.locals["dones"]).item()

		continue_training = self.n_episodes < self._total_max_episodes

		if self.verbose >= 1 and not continue_training:
			mean_episodes_per_env = self.n_episodes / self.training_env.num_envs
			mean_ep_str = (
				f"with an average of {mean_episodes_per_env:.2f} episodes per env" if self.training_env.num_envs > 1 else ""
			)

			print(
				f"Stopping training with a total of {self.num_timesteps} steps because the "
				f"{self.locals.get('tb_log_name')} model reached max_episodes={self.max_episodes}, "
				f"by playing for {self.n_episodes} episodes "
				f"{mean_ep_str}"
			)
		return continue_training


class StopTrainingOnNoModelImprovement(BaseCallback):
	"""
	Stop the training early if there is no new best model (new best mean reward) after more than N consecutive evaluations.

	It is possible to define a minimum number of evaluations before start to count evaluations without improvement.

	It must be used with the ``EvalCallback``.

	:param max_no_improvement_evals: Maximum number of consecutive evaluations without a new best model.
	:param min_evals: Number of evaluations before start to count evaluations without improvements.
	:param verbose: Verbosity level: 0 for no output, 1 for indicating when training ended because no new best model
	"""

	parent: EvalCallback

	def __init__(self, max_no_improvement_evals: int, min_evals: int = 0, rel_improve_tol: float = 0, improve_tol: float = None, verbose: int = 0):
		super().__init__(verbose=verbose)
		self.max_no_improvement_evals = max_no_improvement_evals
		self.min_evals = min_evals
		self.rel_improve_tol = rel_improve_tol
		if (improve_tol is None):
			self.improve_tol = -np.inf
		else:
			self.improve_tol = improve_tol
		self.last_best_mean_reward = -np.inf
		self.no_improvement_evals = 0

	def _on_step(self) -> bool:
		assert self.parent is not None, "``StopTrainingOnNoModelImprovement`` callback must be used with an ``EvalCallback``"

		continue_training = True

		if self.n_calls > self.min_evals:
			if self.parent.best_mean_reward > (self.last_best_mean_reward - self.improve_tol):
				self.no_improvement_evals = 0
			else:
				self.no_improvement_evals += 1
				if self.no_improvement_evals > self.max_no_improvement_evals:
					if (hasattr(self.parent, '_cleanup')):
						self.parent._cleanup()
					continue_training = False

		self.last_best_mean_reward = self.parent.best_mean_reward
		if ((np.isinf(self.improve_tol)) and (not np.isinf(self.last_best_mean_reward))):
			# set tolerance to be a fraction of the initial best reward
			self.improve_tol = self.last_best_mean_reward * self.rel_improve_tol

		if self.verbose >= 1 and not continue_training:
			print(
				f"Stopping training because there was no new best model in the last {self.no_improvement_evals:d} evaluations"
			)

		return continue_training


class ProgressBarCallback(BaseCallback):
	"""
	Display a progress bar when training SB3 agent
	using tqdm and rich packages.
	"""

	pbar: tqdm

	def __init__(self) -> None:
		super().__init__()
		if tqdm is None:
			raise ImportError(
				"You must install tqdm and rich in order to use the progress bar callback. "
				"It is included if you install stable-baselines with the extra packages: "
				"`pip install stable-baselines3[extra]`"
			)

	def _on_training_start(self) -> None:
		# Initialize progress bar
		# Remove timesteps that were done in previous training sessions
		self.pbar = tqdm(total=self.locals["total_timesteps"] - self.model.num_timesteps)

	def _on_step(self) -> bool:
		# Update progress bar, we do num_envs steps per call to `env.step()`
		self.pbar.update(self.training_env.num_envs)
		return True

	def _on_training_end(self) -> None:
		# Flush and close progress bar
		self.pbar.refresh()
		self.pbar.close()


class CustomSaveLogCallback(BaseCallback):
	"""
	A callback to periodically save the model that derives from ``BaseCallback``.

	:param verbose: (int) Verbosity level 0: not output 1: info 2: debug
	"""
	def __init__(self, save_every_timestep, save_path, save_prefix='model', verbose=0, termination=None):
		super(CustomSaveLogCallback, self).__init__(verbose)
		self.save_every_timestep = save_every_timestep

		assert os.path.isdir(save_path), 'Save directory does not exist!'
		self.save_path = save_path
		self.save_prefix = save_prefix
		self.curr_check_point_id = 0
		if type(termination)==dict:
			self.termination_criteria = termination.get('criteria', 'reward')
			self.termination_threshold = termination.get('threshold', 0.025)
			self.termination_criteria_count = 0
			self.termination_repeat = termination.get('repeat', 10)

	def _on_training_start(self) -> None:
		pass

	def _on_rollout_start(self) -> None:
		pass

	def _on_step(self) -> bool:
		# log terminal episodic info
		terminal_infos = [info_i for ii, info_i in enumerate(self.locals['infos']) if self.locals['dones'][ii]]
		if (len(terminal_infos) > 0):
			terminal_statnames = [ts for ts in terminal_infos[0].keys() if (ts.startswith('ep_'))]
			for ts in terminal_statnames:
				ts_stat = [terminal_infos[ii][ts] for ii in range(len(terminal_infos)) if (not np.isnan(terminal_infos[ii][ts]))]
				self.logger.record("rollout/" + ts, np.mean(ts_stat))

		# save model
		check_point_id = divmod(self.model.num_timesteps+1, self.save_every_timestep)[0]
		if (check_point_id > self.curr_check_point_id):
			self.curr_check_point_id = check_point_id
			save_id = check_point_id*self.save_every_timestep
			self.model.save(os.path.join(self.save_path, self.save_prefix + '_' + str(save_id)))

		continue_training = True
		if (hasattr(self, "termination_criteria_count") and hasattr(self, "termination_repeat") and (self.termination_criteria_count >= self.termination_repeat)):
			continue_training = False

		return continue_training

	def _on_rollout_end(self) -> None:
		if (hasattr(self, "termination_criteria") and hasattr(self.model, 'ep_info_buffer')):
			if (self.termination_criteria == "reward"):
				mean_criteria = np.mean([info["r"] for info in self.model.ep_info_buffer if (("r" in info.keys()) and (not np.isnan(info["r"])))])
			else:
				NotImplementedError

			if (hasattr(self, "mean_criteria_last")):
				delta = mean_criteria - self.mean_criteria_last
				self.logger.record('rollout/delta_termination_criteria', delta)
				self.mean_criteria_last = mean_criteria
				if (np.abs(delta) < self.termination_threshold):
					self.termination_criteria_count += 1
				elif (delta > self.termination_threshold):
					self.termination_criteria_count = 0
			else:
				self.mean_criteria_last = mean_criteria
				self.termination_criteria_count = 0	
			self.logger.record('rollout/termination_criteria_count', self.termination_criteria_count)

	def _on_training_end(self) -> None:
		pass
