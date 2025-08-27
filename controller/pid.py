import numpy as np

class PID():
    def __init__(self,Ts=0.1):
        self._Ts = Ts
        self._e_prev = 0
        self._e_int = 0

    def control(self,ref,x):
        Kp = 10.
        Ki = 0.5
        Kd = 0.1

        e = ref - x

        P = Kp * e
        I = Ki * self._e_int
        D = Kd * (e - self._e_prev)

        self._e_prev = e
        self._e_int += e

        return P+I+D