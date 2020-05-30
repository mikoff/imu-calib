import numpy as np

from code.imu_model import skew

class Quaternion():
    '''
    Quaternion class. Supports multiplication, normalization, conversion to matrix form.
            Notation: right-handed, passive, local-to-global.
    '''
    def __init__(self, q = [1, 0, 0, 0]):
        self._q = np.array(q).reshape((4,1))
        
    @property
    def q(self):
        return self._q
    
    @q.setter
    def q(self, value):
        self._q = value
    
    @property
    def r(self):
        return self._q[0,0]
    
    @property
    def w(self):
        return self._q[0,0]
    
    @property
    def x(self):
        return self._q[1,0]
    
    @property
    def y(self):
        return self._q[2,0]
    
    @property
    def z(self):
        return self._q[3,0]

    @property
    def v(self):
        return self._q[1:,0]

    @classmethod
    def fromRV(cls, r, v):
        return Quaternion([r, v[0], v[1], v[2]])
    
    @classmethod
    def exactFromOmega(cls, w):
        wnorm = np.linalg.norm(w)
        if (wnorm > 0.00001):
            return Quaternion.fromRV(np.cos(wnorm*0.5), (w / wnorm * np.sin(wnorm*0.5)).flatten())
        else:
            return Quaternion.identity()

    @classmethod
    def from_euler(cls, roll, pitch, yaw):
        halfRoll = roll / 2.0
        halfPitch = pitch / 2.0
        halfYaw = yaw / 2.0
        w = np.cos(halfRoll) * np.cos(halfPitch) * np.cos(halfYaw) + np.sin(halfRoll) * np.sin(halfPitch) * np.sin(halfYaw)
        x = np.sin(halfRoll) * np.cos(halfPitch) * np.cos(halfYaw) - np.cos(halfRoll) * np.sin(halfPitch) * np.sin(halfYaw)
        y = np.cos(halfRoll) * np.sin(halfPitch) * np.cos(halfYaw) + np.sin(halfRoll) * np.cos(halfPitch) * np.sin(halfYaw)
        z = np.cos(halfRoll) * np.cos(halfPitch) * np.sin(halfYaw) - np.sin(halfRoll) * np.sin(halfPitch) * np.cos(halfYaw)
        return Quaternion([w, x, y, z])
    
    @classmethod
    def identity(cls):
        return Quaternion([1, 0, 0, 0])

    def norm(self):
        return np.linalg.norm(self._q)
    
    def conj(self):
        return Quaternion.fromRV(self.r, -1 * self.v)

    def prod(self, other):
        return Quaternion.fromRV(self.r * other.r - self.v.dot(other.v),
                                 self.r * other.v + other.r * self.v + np.cross(self.v, other.v))
    
    def Rm(self):
        return (self.w**2 - self.v.T @ self.v) * np.eye(3)\
                + 2 * np.outer(self.v, self.v.T) + 2 * self.w * skew(self.v)
    
    def normalize(self):
        norm = np.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)
        self.q[0,0] /= norm
        self.q[1,0] /= norm
        self.q[2,0] /= norm
        self.q[3,0] /= norm

    def normalized(self):
        norm = np.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)
        return Quaternion(self.q / norm)

    def to_euler(self):
        q = self.q
        t0 = 2.0 * (q[0] * q[1] + q[2] * q[3])
        t1 = 1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2])
        roll = np.arctan2(t0, t1)[0]
        t2 = 2.0 * (q[0] * q[2] - q[3] * q[1])
        t2 = 1.0 if t2 > 1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch = np.arcsin(t2)[0]
        t3 = 2.0 * (q[0] * q[3] + q[1] * q[2])
        t4 = 1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3])
        yaw = np.arctan2(t3, t4)[0]
        return (roll, pitch, yaw)
