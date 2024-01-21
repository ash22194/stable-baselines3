from gymnasium.envs.registration import register

register(id='Unicycle-v0', entry_point='stable_baselines3.systems.unicycle:Unicycle')
# register(id='UnicycleSkipStep-v0', entry_point='stable_baselines3.systems.unicycle_skipstep:UnicycleSkipStep')

register(id='Quadcopter-v0', entry_point='stable_baselines3.systems.quadcopter:Quadcopter')
register(id='QuadcopterSkipStep-v0', entry_point='stable_baselines3.systems.quadcopter_skipstep:QuadcopterSkipStep')
register(id='QuadcopterTT-v0', entry_point='stable_baselines3.systems.quadcopter_tt:QuadcopterTT')
# register(id='QuadcopterStateOutput-v0', entry_point='stable_baselines3.systems.quadcopter_stateoutput:QuadcopterStateOutput')

register(id='CartPoleCustom-v0', entry_point='stable_baselines3.systems.cartpole:CartPole')

# register(id='Manipulator4DOF-v0',
# 		 entry_point='systems.manipulator4dof:Manipulator4DOF')

# register(id='Biped2D-v0',
# 		 entry_point='systems.biped2d:Biped2D')