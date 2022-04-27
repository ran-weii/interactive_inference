import numpy as np

class ConstantAcceleration:
    def __init__(self, dt=0.1):
        """
        Constant acceleration dynamics model with state variables
            [x, y, xv, xy, ax, ay]
        """
        self.dt = dt

        self.A = np.array(
            [[1, 0, dt,  0,  0.5 * dt**2,  0],
            [0, 1,  0, dt,  0,  0.5 * dt**2],
            [0, 0,  1,  0, dt,  0],
            [0, 0,  0,  1,  0, dt],
            [0, 0,  0,  0,  1,  0],
            [0, 0,  0,  0,  0,  1]]
        ) 

    def step(self, state):
        next_state = np.matmul(self.A, state)
        return next_state