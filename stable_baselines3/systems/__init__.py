from gymnasium.envs.registration import register

register(id='Unicycle-v0', entry_point='stable_baselines3.systems.unicycle:Unicycle')
# register(id='UnicycleSkipStep-v0', entry_point='stable_baselines3.systems.unicycle_skipstep:UnicycleSkipStep')
register(id='UnicycleTT-v0', entry_point='stable_baselines3.systems.unicycle_tt:UnicycleTT')
register(id='UnicycleDecomposition-v0', entry_point='stable_baselines3.systems.unicycle_decomposition:UnicycleDecomposition')

register(id='Quadcopter-v0', entry_point='stable_baselines3.systems.quadcopter:Quadcopter')
register(id='QuadcopterSkipStep-v0', entry_point='stable_baselines3.systems.quadcopter_skipstep:QuadcopterSkipStep')
register(id='QuadcopterTT-v0', entry_point='stable_baselines3.systems.quadcopter_tt:QuadcopterTT')
register(id='QuadcopterDecomposition-v0', entry_point='stable_baselines3.systems.quadcopter_decomposition:QuadcopterDecomposition')
register(id='QuadcopterTTDecomposition-v0', entry_point='stable_baselines3.systems.quadcopter_tt_decomposition:QuadcopterTTDecomposition')

register(id='CartPoleCustom-v0', entry_point='stable_baselines3.systems.cartpole:CartPole')

# register(id='Manipulator4DOF-v0',
# 		 entry_point='systems.manipulator4dof:Manipulator4DOF')

# register(id='Biped2D-v0',
# 		 entry_point='systems.biped2d:Biped2D')