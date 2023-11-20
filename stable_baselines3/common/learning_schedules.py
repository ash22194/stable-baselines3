import numpy as np
from typing import Callable

def exponential_schedule(initial_value: float, decay_rate=10) -> Callable[[float], float]:
	"""
	Exponential schedule.

	:param initial_value: Initial learning rate.
	:param decay_rate in fraction of progress
	:return: schedule that computes
	  current learning rate depending on remaining progress
	"""
	def func(progress_remaining: float) -> float:
		"""
		Progress will decrease from 1 (beginning) to 0.

		:param progress_remaining:
		:return: current learning rate
		"""
		return (initial_value*np.exp(-decay_rate*(1 - progress_remaining)))

	return func

def decay_sawtooth_schedule(initial_value: float, sawtooth_width=0.1) -> Callable[[float], float]:
	"""
	Decaying sawtooth rate schedule.

	:param initial_value: Initial learning rate.
	:param sawtooth width in fraction of progress
	:return: schedule that computes
	  current learning rate depending on remaining progress
	"""
	def func(progress_remaining: float) -> float:
		"""
		Progress will decrease from 1 (beginning) to 0.

		:param progress_remaining:
		:return: current learning rate
		"""
		d, r = divmod(1 - progress_remaining, sawtooth_width)
		return (initial_value/(d+1)*(1 - (r/sawtooth_width)))

	return func

def linear_schedule(initial_value: float) -> Callable[[float], float]:
	"""
	Linear learning rate schedule.

	:param initial_value: Initial learning rate.
	:return: schedule that computes
	  current learning rate depending on remaining progress
	"""
	def func(progress_remaining: float) -> float:
		"""
		Progress will decrease from 1 (beginning) to 0.

		:param progress_remaining:
		:return: current learning rate
		"""
		return progress_remaining * initial_value

	return func