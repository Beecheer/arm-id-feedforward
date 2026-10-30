"""Verify moved data provenance and shallow recording integrity."""
import json
from pathlib import Path
import tempfile
import unittest

from arm_id_feedforward.arm_datasets import flat_recording, resolve_provenance, sha256


class DatasetTests(unittest.TestCase):
    def test_resolve_moved_snapshot_by_hash_without_rewriting_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);snapshot=root/'results'/'frozen.json';snapshot.parent.mkdir()
            payload=b'{"capture":"/old/deep/path/capture.csv","parameters":[1,2,3]}\n'
            snapshot.write_bytes(payload);digest=sha256(snapshot)
            index=root/'datasets.json'
            index.write_text(json.dumps(dict(kind='arm_dataset_catalog',version=1,files=[dict(path='results/frozen.json',sha256=digest)])))
            actual=resolve_provenance('/removed/deep/frozen.json',digest,index)
            self.assertEqual(actual,snapshot)
            self.assertEqual(actual.read_bytes(),payload)
            snapshot.write_bytes(b'changed')
            with self.assertRaises(FileNotFoundError):resolve_provenance('/removed/deep/frozen.json',digest,index)

    def test_flat_recording_checks_capture_and_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);capture=root/'train.csv';plan=root/'train_plan.csv'
            capture.write_text('original feedback');plan.write_text('original plan')
            digest=sha256(capture)
            (root/'train.json').write_text(json.dumps(dict(trajectory_sha256=sha256(plan))))
            (root/'train_quality.json').write_text(json.dumps(dict(capture_sha256=digest,issues=[])))
            (root/'dataset.json').write_text(json.dumps(dict(kind='arm_recording_group',version=1,records={
                'dynamic':dict(capture='train.csv',plan='train_plan.csv',quality='train_quality.json',sha256=digest)})))
            self.assertEqual(flat_recording(root,'dynamic')[0],capture)
            plan.write_text('changed plan')
            with self.assertRaises(ValueError):flat_recording(root,'dynamic')
            plan.write_text('original plan');capture.write_text('changed feedback')
            with self.assertRaises(ValueError):flat_recording(root,'dynamic')


if __name__=='__main__':unittest.main()
