import numpy as np

class ConstantAcceleration:
    def __init__(self, dt=0.1):
        """
        Constant acceleration dynamics model with state and action variables
            [x, y, xv, xy], [ax, ay]
        """
        self.dt = dt

        # self.A = np.array(
        #     [[1, 0, dt,  0,  0.5 * dt**2,  0],
        #     [0, 1,  0, dt,  0,  0.5 * dt**2],
        #     [0, 0,  1,  0, dt,  0],
        #     [0, 0,  0,  1,  0, dt],
        #     [0, 0,  0,  0,  1,  0],
        #     [0, 0,  0,  0,  0,  1]]
        # ) 

        self.A = np.array(
            [[1, 0, dt,  0],
            [0, 1,  0, dt],
            [0, 0,  1,  0],
            [0, 0,  0,  1]]
        ) 
        
        self.B = np.array(
            [[0.5 * dt**2, 0],
            [0, 0.5 * dt**2],
            [dt, 0],
            [0, dt]]
        )
    def step(self, state, action):
        """
        Args:
            state (np.array): size=[batch_size, 4]
            action (np.array): size=[batch_size, 2]

        Returns:
            next_state (np.array): size=[batch_size, 4]
        """
        # next_state = np.matmul(self.A, state)
        next_state = np.matmul(self.A, state) + np.matmul(self.B, action)
        return next_state