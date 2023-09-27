from gym.envs.registration import register

register(id='Quadcopter-v0',
		 entry_point='systems.quadcopter:Quadcopter')

register(id='Manipulator4DOF-v0',
		 entry_point='systems.manipulator4dof:Manipulator4DOF')

register(id='Biped2D-v0',
		 entry_point='systems.biped2d:Biped2D')