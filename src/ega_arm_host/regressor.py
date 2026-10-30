"""The 91-column shared inverse-dynamics regressor used by host control.

Rows are joint-major. The first 70 columns are Pinocchio's ten inertial
parameters per link, followed by seven viscous, Coulomb and rotor terms each.
"""

from __future__ import annotations

import numpy as np
import pinocchio as pin

from . import arm_collect


class DynamicRegressor:
    def __init__(self):
        self.model = arm_collect._load_urdf(arm_collect.URDF)
        self.data = self.model.createData()

    def matrix(self, q, v, a):
        q, v, a = (np.asarray(x, dtype=float) for x in (q, v, a))
        if q.ndim != 2 or q.shape[1] != 7 or q.shape != v.shape or q.shape != a.shape:
            raise ValueError("regressor inputs must have matching N×7 shapes")
        samples = len(q)
        result = np.zeros((samples * 7, 91), dtype=float)
        for sample in range(samples):
            rigid = pin.computeJointTorqueRegressor(
                self.model, self.data, q[sample], v[sample], a[sample]
            )
            for joint in range(7):
                row = joint * samples + sample
                result[row, :70] = rigid[joint]
                result[row, 70 + joint] = v[sample, joint]
                result[row, 77 + joint] = np.sign(v[sample, joint])
                result[row, 84 + joint] = a[sample, joint]
        return result
