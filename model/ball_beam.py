import numpy as np
import torch.nn as nn

class BallAndBeam():
    def __init__(self, ms=0.1, Ib=0.1875, g=9.8):
        self._g = g
        self._ms = ms
        self._Ib = Ib

    def f(self,x,u):
        g = self._g
        ms = self._ms
        Ib = self._Ib

        xdot = np.zeros(4)
        xdot[0] = x[1]
        xdot[1] = -(5/7) * x[0] * (x[3]**2) - (5/7) * g * np.sin(x[2])
        xdot[2] = x[3]
        xdot[3] = -(2*ms*x[0]*x[1]*x[3])/(Ib+ms*(x[0]**2)) - \
                (ms*g*x[0]*np.cos(x[2]))/(Ib+ms*(x[0]**2)) + \
                u / (Ib + ms * (x[0]**2))
        
        return xdot
    
    def linearized_model(self):
        """
        Linearized of ball and beam model. Assume x->0 and theta-> 0 \n
        In general written as `xdot = Ax + Bu`\n
        Input: None \n
        Output: `A`,`B`
        """
        g = 9.8
        ms = 0.1
        Ib = 0.1875

        A = np.array([
            [0, 1, 0, 0],
            [0, 0, -(5/7)*g, 0],
            [0, 0, 0, 1],
            [-ms*g/Ib, 0, 0, 0]
        ])

        B = np.array([
            [0],
            [0],
            [0],
            [1 / Ib]
        ])
        return A,B
    
class BallBeamNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )

    def forward(self, input):
        output = self.net(input)
        return output