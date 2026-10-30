"""Supplemental coverage and collection sequencing, without USB."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from ega_arm_host import arm_collect_more as more
from ega_arm_host import arm_dynamics
from ega_arm_host import arm_excitation


class SupplementalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dynamics = arm_dynamics.DynamicModel()
        cls.basis, _ = cls.dynamics.reference_basis()
        cls.rows = more.row_bases(cls.dynamics, cls.basis)

    def test_independent_reference_and_anchored_initials(self):
        self.assertEqual([T.shape[1] for T in self.rows], [42,44,38,31,24,17,10])
        screen = MagicMock()
        screen.clear.return_value = True
        grams = [np.zeros((T.shape[1], T.shape[1])) for T in self.rows]
        for targets in more.TARGETS.values():
            for target in targets:
                problem = more.CoverageProblem(self.dynamics, self.basis, screen, 71, None,
                    target=target, rows=self.rows, grams=grams)
                problem.check_candidate(problem.initial)
                t, q, v, a = problem.motion.evaluate(problem.initial, 100.)
                self.assertEqual(t[-1], 32.)
                self.assertLess(abs(v).max(), arm_excitation.MAX_VELOCITY)
                self.assertLess(abs(a).max(), arm_excitation.MAX_ACCELERATION)
                np.testing.assert_allclose(v[[0,-1]], 0, atol=1e-12)
                np.testing.assert_allclose(a[[0,-1]], 0, atol=1e-12)
                with patch('ega_arm_host.arm_excitation.pin.updateGeometryPlacements'), patch('ega_arm_host.arm_excitation.pin.computeDistances'):
                    screen.geometry_data.distanceResults = []
                    screen.geometry.collisionPairs = []
                    self.assertEqual(len(problem.constraints(problem.initial)), len(problem.get_constraint_bounds()[0]))
                with self.assertRaises(ValueError):
                    problem.check_candidate(np.tile(more.arm_plan.HOME, (5,1)).ravel())

    def test_information_accumulates_only_after_accepting_a_segment(self):
        screen = MagicMock(); screen.clear.return_value = True
        grams = [np.zeros((T.shape[1], T.shape[1])) for T in self.rows]
        problem = more.CoverageProblem(self.dynamics, self.basis, screen, 71, None,
            target=more.TARGETS['train_01'][0], rows=self.rows, grams=grams)
        original = [G.copy() for G in grams]
        blocks = problem.blocks(problem.initial)
        problem._objective(problem.initial)
        for G, before in zip(grams, original): np.testing.assert_array_equal(G, before)
        problem.on_selected(problem.initial)
        for G, W in zip(grams, blocks): np.testing.assert_allclose(G, W.T@W)
        problem.on_selected(problem.initial)
        for G, W in zip(grams, blocks): np.testing.assert_allclose(G, 2*W.T@W)

    def test_collect_resumes_without_identifying_or_repeating_completed_run(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output/'pack.json').write_text(json.dumps(dict(kind='arm_supplemental_collection',version=1)))
            workflows = []
            def workflow(path, port):
                value = MagicMock()
                value.output = path
                value.state = dict(completed=['dynamic'] if path.name=='train_01' else [])
                value.port = 'fixture'
                workflows.append(value)
                return value
            with patch.object(more,'confirm'), patch.object(more,'Workflow',side_effect=workflow):
                more.collect(output)
            self.assertFalse(workflows[0].run_stage.called)
            for item in workflows[1:3]: item.run_stage.assert_called_once_with('dynamic')
            workflows[3].run_stage.assert_called_once_with('validation')

    def test_incomplete_run_can_be_recollected_without_deleting_old_files(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)
            (output/'pack.json').write_text(json.dumps(dict(kind='arm_supplemental_collection',version=1)))
            first=output/'train_01'/'dynamic'/'run_001';first.mkdir(parents=True)
            (first/'capture.csv').write_text('original data')
            (first/'quality_report.json').write_text(json.dumps({'issues':['动态轨迹采集时间覆盖不完整']}))
            workflows=[]
            def workflow(path,port):
                value=MagicMock();value.output=path;value.port='fixture'
                value.state={'completed':['dynamic' if path.name!='blind' else 'validation'],
                    'attempts':{'dynamic':{'directory':'dynamic/run_001','phase':'complete'}}}
                workflows.append(value);return value
            with patch.object(more,'confirm'), patch('builtins.input',return_value='y'), \
                 patch.object(more,'Workflow',side_effect=workflow): more.collect(output)
            workflows[0].run_stage.assert_called_once_with('dynamic')
            self.assertNotIn('dynamic',workflows[0].state['completed'])
            self.assertEqual((first/'capture.csv').read_text(),'original data')

    def test_cancel_before_collection_does_not_open_hardware(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output/'pack.json').write_text(json.dumps(dict(kind='arm_supplemental_collection',version=1)))
            with patch.object(more,'confirm',side_effect=more.Cancelled), patch.object(more,'Workflow') as workflow:
                with self.assertRaises(more.Cancelled): more.collect(output)
                workflow.assert_not_called()


if __name__ == '__main__': unittest.main()
