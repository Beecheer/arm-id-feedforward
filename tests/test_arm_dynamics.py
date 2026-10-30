"""Dynamic basis, derivative handling and trajectory shape checks; no USB."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import numpy as np
import pinocchio as pin

from ega_arm_host import arm_collect
from ega_arm_host import arm_dynamics
from ega_arm_host import arm_excitation


class DynamicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dynamics=arm_dynamics.DynamicModel()

    def test_figaroh_parameter_layout_matches_dynamics_and_friction(self):
        rng=np.random.default_rng(3)
        q=rng.normal(size=(10,7));v=rng.normal(size=(10,7));a=rng.normal(size=(10,7))
        W=self.dynamics.matrix(q,v,a)
        theta=np.concatenate([x.toDynamicParameters() for x in self.dynamics.model.inertias[1:]])
        parameters=np.r_[theta,np.full(7,.02),np.full(7,.04),np.full(7,.003)]
        pred=(W@parameters).reshape(7,-1).T
        data=self.dynamics.model.createData()
        expected=np.array([pin.rnea(self.dynamics.model,data,qq,vv,aa).copy() for qq,vv,aa in zip(q,v,a)])
        expected+=.02*v+.04*np.sign(v)+.003*a
        np.testing.assert_allclose(pred,expected,atol=1e-12)

    def test_reference_basis_has_consistent_full_rank(self):
        transform,report=self.dynamics.reference_basis()
        self.assertEqual(transform.shape,(91,62))
        self.assertEqual(report['rank'],62)
        rng=np.random.default_rng(4)
        q=rng.normal(size=(100,7));v=rng.normal(size=(100,7));a=rng.normal(size=(100,7))
        W=self.dynamics.matrix(q,v,a)
        target=W@rng.normal(size=91)
        coefficient=arm_dynamics.ridge_fit(W@transform,target,0.)
        np.testing.assert_allclose(W@transform@coefficient,target,atol=1e-9)

    def test_spline_endpoints_and_analytic_derivatives(self):
        motion=arm_excitation.SplineMotion()
        x=np.tile([0,.9,.1,-.8,.2,.25,0],(5,1)).reshape(-1)
        t,q,v,a=motion.evaluate(x,100.)
        np.testing.assert_allclose(q[[0,-1]],np.tile(arm_excitation.arm_plan.HOME,(2,1)),atol=1e-12)
        np.testing.assert_allclose(v[[0,-1]],0,atol=1e-12)
        np.testing.assert_allclose(a[[0,-1]],0,atol=1e-12)
        np.testing.assert_allclose(np.gradient(q,t,axis=0)[2:-2],v[2:-2],atol=1e-5)

    def test_processing_recovers_sine_derivatives_and_does_not_bridge_gaps(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'capture.csv'
            times=np.r_[np.arange(0,8,.005),np.arange(9,17,.005)]
            q=np.sin(2*np.pi*.2*times)
            with path.open('w',newline='') as stream:
                writer=csv.writer(stream)
                writer.writerow(['device_time_us','sample_sequence','status',*arm_dynamics.Q,*arm_dynamics.TAU])
                for i,t in enumerate(times):writer.writerow([round(t*1e6),i,'0x000f',*([q[i]]*7),*([.3*q[i]]*7)])
            path.with_suffix('.json').write_text(json.dumps({'transition_duration_s':0.}))
            (t,qq,v,a,tau),report=arm_dynamics.process_capture(path)
        self.assertEqual(len(report['segments']),2)
        self.assertEqual(report['feedback_gaps_or_resets'],1)
        np.testing.assert_allclose(v[:,0],2*np.pi*.2*np.cos(2*np.pi*.2*t),atol=.003)
        np.testing.assert_allclose(a[:,0],-(2*np.pi*.2)**2*np.sin(2*np.pi*.2*t),atol=.015)
        self.assertFalse(np.any((t>7.1)&(t<9.9)))

    def test_excitation_bounds_do_not_invoke_torque_screen(self):
        basis,_=self.dynamics.reference_basis()
        screen=MagicMock();screen.clear.return_value=True
        with patch('ega_arm_host.arm_plan.TorqueScreen',side_effect=AssertionError('no torque constraints')):
            problem=arm_excitation.ExcitationProblem(self.dynamics,basis,screen,5)
        lower,upper=map(np.array,problem.get_variable_bounds())
        self.assertTrue(np.all(lower.reshape(-1,7)[:,1]>=np.radians(30)-1e-10))
        self.assertTrue(np.all(upper.reshape(-1,7)[:,3]<=-np.radians(30)+1e-5))

    def test_end_to_end_fit_predicts_separate_synthetic_trajectory(self):
        rng=np.random.default_rng(16)
        theta=np.r_[np.concatenate([i.toDynamicParameters() for i in self.dynamics.model.inertias[1:]]),
                    np.full(7,.02),np.full(7,.04),np.full(7,.003)]
        def generate():
            parts=[]
            motion=arm_excitation.SplineMotion()
            for index in range(3):
                x=np.tile(arm_excitation.arm_plan.HOME,(5,1))+rng.uniform(-.5,.5,(5,7))
                t,q,v,a=motion.evaluate(x.reshape(-1),5.)
                parts.append((t[:-1]+index*24,q[:-1],v[:-1],a[:-1]))
            t,q,v,a=[np.concatenate([p[k] for p in parts]) for k in range(4)]
            tau=(self.dynamics.matrix(q,v,a)@theta).reshape(7,-1).T
            return (t,q,v,a,tau),{'processed_samples':len(t),'synthetic':True}
        train=generate();valid=generate()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);capture=root/'train.csv';validation=root/'validation.csv'
            capture.write_text('synthetic_train');validation.write_text('synthetic_validation')
            with patch('ega_arm_host.arm_dynamics.process_capture',side_effect=lambda p,c:train if p==capture else valid):
                report=arm_dynamics.fit_capture(capture,root/'result',validation)
            self.assertTrue(report['independent_validation_available'])
            self.assertLess(max(report['validation']['rmse_nm']),1e-7)
            self.assertTrue((root/'result/dynamic_model.json').exists())


if __name__=='__main__':unittest.main()
