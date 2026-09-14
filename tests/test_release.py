"""CPU-only contracts: no trainer import, GPU use, networking, or process kills."""
from __future__ import annotations
import ast
import copy
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'deploy'))
import release_lib as lib
import gb10_apply as gpu
import pull_release as pull

class SourceContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = ROOT / 'trainer/agillm44.py'
        cls.tree = ast.parse(cls.path.read_text())
    def test_source_and_embedded_modules_compile(self):
        report = lib.source_checks(self.path)
        self.assertGreaterEqual(report['folded_modules_compiled'], 4)
    def test_manifest_matches_canonical_source(self):
        self.assertEqual(lib.validate_release(ROOT)['source_sha256'], lib.digest(self.path))
    def test_strict_json_nonfinite_regression(self):
        function = next(n for n in self.tree.body if isinstance(n, ast.FunctionDef) and n.name == '_agillm43_sanitize_json_floats')
        namespace = {}; exec(compile(ast.Module(body=[function], type_ignores=[]), '<isolated-sanitizer>', 'exec'), namespace)
        original = {'nested': [float('nan'), {'infinity': float('inf')}], 'tuple': (float('-inf'), 3.0), 'valid': 4.0}
        cleaned = namespace[function.name](original)
        self.assertEqual(json.loads(json.dumps(cleaned, allow_nan=False)), {'nested': [None, {'infinity': None}], 'tuple': [None, 3.0], 'valid': 4.0})
        self.assertTrue(math.isnan(original['nested'][0]))
    def test_digest_mismatch_is_rejected(self):
        with patch.object(lib, 'source_checks', return_value={'sha256': '0'*64}):
            with self.assertRaisesRegex(ValueError, 'digest mismatch'): lib.validate_release(ROOT)

class RuntimeContracts(unittest.TestCase):
    def setUp(self):
        self.argv = ['python3', 'agillm44.py', 'train', '--require_exact_resume_state', '--dblock', '--dblock_ar_prob', '0.8', '--dblock_sat_prob', '0.1', '--dblock_nat_prob', '0.1', '--sat_every', '1', '--nat_every', '1', '--dblock_direct56_enabled', '1', '--resume', '/checkpoint.pt']
        self.hot = {'dblock_satvar_enabled': 1, 'owner_satvar_must_choose': True, 'dblock_satvar_block_probs': '1:0.5,2:0.5'}
    def test_three_modes_and_dynamic_satvar_preserved(self): lib.validate_runtime(self.argv, self.hot)
    def test_fixed_or_disabled_satvar_is_rejected(self):
        for value in (0, False, None):
            with self.subTest(value=value):
                with self.assertRaises((ValueError, TypeError)): lib.validate_runtime(self.argv, {'dblock_satvar_enabled': value})
    def test_single_stride_distribution_rejected(self):
        for distribution in ('1:1.0', '2:1.0', '1:0,2:1', '1:1,2:0'):
            hot = dict(self.hot, dblock_satvar_block_probs=distribution)
            with self.assertRaises(ValueError): lib.validate_runtime(self.argv, hot)
    def test_owner_choose_contract_cannot_be_removed(self):
        hot = dict(self.hot); hot.pop('owner_satvar_must_choose')
        with self.assertRaises(ValueError): lib.validate_runtime(self.argv, hot)
    def test_each_disabled_mode_is_rejected(self):
        for mode in ('ar', 'sat', 'nat'):
            args = lib.replace_argument(self.argv, '--dblock_' + mode + '_prob', '0')
            with self.assertRaises(ValueError): lib.validate_runtime(args, self.hot)
    def test_nonfinite_mode_probability_is_rejected(self):
        with self.assertRaises(ValueError): lib.validate_runtime(lib.replace_argument(self.argv, '--dblock_ar_prob', 'nan'), self.hot)
    def test_optimizer_and_lr_reset_flags_are_rejected(self):
        for flag in ('--fresh', '--reset_optimizer_on_resume', '--lr_schedule_reset_on_resume', '--lr_schedule_reanchor_on_resume'):
            with self.subTest(flag=flag):
                with self.assertRaises(ValueError): lib.validate_runtime(self.argv + [flag], self.hot)
    def test_duplicate_arguments_are_rejected(self):
        with self.assertRaises(ValueError): lib.argument(self.argv + ['--resume', 'other.pt'], '--resume')
    def test_replace_preserves_unrelated_args(self):
        updated = lib.replace_argument(self.argv, '--resume', 'new.pt')
        self.assertEqual(updated[:-1], self.argv[:-1]); self.assertEqual(updated[-1], 'new.pt')
    def test_missing_exact_resume_rejected(self):
        with self.assertRaises(ValueError): lib.validate_runtime([x for x in self.argv if x != '--require_exact_resume_state'], self.hot)

class CheckpointContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); name = 'checkpoint.pt'; shard = name + '.shards'
        (self.root / shard).mkdir()
        shard_names = ['core_base.pt', 'ar.pt', 'sat.pt', 'nat.pt', 'opt.pt', 'scaler.pt', 'detachable_50m_base.pt', 'agillm5_detachable_50m_bridge.pt', 'agillm5_detachable_50m_opt.pt', 'agillm5_detachable_50m_scaler.pt', 'agillm5_detachable_50m_rng.pt', 'agillm5_detachable_50m_counters.pt'] + ['core_block_%04d.pt' % i for i in range(28)] + ['detachable_50m_block_%04d.pt' % i for i in range(22)]
        for rel in (name, name + '.tokenizer.json', *[shard + '/' + x for x in shard_names]):
            (self.root / rel).write_text('safe fixture: ' + rel)
        def row(path): return {'name': path.name, 'sha256': lib.digest(path), 'nbytes': path.stat().st_size}
        main = row(self.root / name); tok = row(self.root / (name + '.tokenizer.json'))
        self.manifest = {'schema': 'agillm43.training.resume.v2', 'role': 'training', 'step': 100, 'seen_tok': 1000, 'checkpoint_name': name, 'shard_dir': shard, 'checkpoint_main_sha256': main['sha256'], 'checkpoint_main_nbytes': main['nbytes'], 'tokenizer_sidecar_sha256': tok['sha256'], 'tokenizer_sidecar_nbytes': tok['nbytes'], 'shards': [row(p) for p in sorted((self.root / shard).iterdir())]}
        self.save()
    def save(self): lib.atomic_json(self.root / 'training_latest.json', self.manifest)
    def test_complete_checkpoint_accepted(self): self.assertEqual(lib.verified_checkpoint(self.root, 100)[0]['step'], 100)
    def test_stale_checkpoint_rejected(self):
        with self.assertRaises(ValueError): lib.verified_checkpoint(self.root, 101)
    def test_serving_pointer_rejected(self):
        self.manifest['role'] = 'serving'; self.save()
        with self.assertRaises(ValueError): lib.verified_checkpoint(self.root, 100)
    def test_corrupt_shard_rejected(self):
        p = self.root / 'checkpoint.pt.shards/ar.pt'; p.write_text('x' * p.stat().st_size)
        with self.assertRaisesRegex(ValueError, 'hash mismatch'): lib.verified_checkpoint(self.root, 100)
    def test_missing_optimizer_rejected(self):
        self.manifest['shards'] = [r for r in self.manifest['shards'] if r['name'] != 'opt.pt']; self.save()
        with self.assertRaises(ValueError): lib.verified_checkpoint(self.root, 100)
    def test_path_traversal_rejected(self):
        self.manifest['shards'][0]['name'] = '../outside'; self.save()
        with self.assertRaisesRegex(ValueError, 'unsafe'): lib.verified_checkpoint(self.root, 100)
    def test_symlink_shard_rejected(self):
        p = self.root / 'checkpoint.pt.shards/ar.pt'; content = p.read_bytes(); p.unlink()
        outside = self.root / 'other'; outside.write_bytes(content); p.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, 'unsafe'): lib.verified_checkpoint(self.root, 100)
    def test_duplicate_shards_rejected(self):
        self.manifest['shards'].append(self.manifest['shards'][0]); self.save()
        with self.assertRaisesRegex(ValueError, 'duplicate'): lib.verified_checkpoint(self.root, 100)
    def test_pin_survives_original_file_deletion(self):
        _, p = gpu.pin_checkpoint(self.root, self.root / 'pinned', 100)
        (self.root / 'checkpoint.pt').unlink()
        self.assertTrue(p.exists()); lib.verified_checkpoint(p.parent, 100)

class DeploymentContracts(unittest.TestCase):
    def test_changed_pid_never_signalled(self):
        with patch.object(gpu, 'same_process', return_value=False), patch.object(gpu.os, 'pidfd_open') as opened:
            with self.assertRaises(ValueError): gpu.graceful_stop({'pid': 123})
            opened.assert_not_called()
    def test_existing_trainer_prevents_second_launch(self):
        with patch.object(gpu, 'trainers', return_value=[{'pid': 123}]), patch.object(gpu.subprocess, 'Popen') as start:
            with self.assertRaises(ValueError): gpu.launch({}, Path('/a.py'), Path('/a.pt'), 'test')
            start.assert_not_called()
    def test_identical_source_binds_without_stop_or_launch(self):
        manifest = lib.validate_release(ROOT)
        live = {'pid': 123, 'source': '/existing/agillm44.py', 'source_sha256': manifest['source_sha256']}
        with tempfile.TemporaryDirectory() as tmp, patch.object(gpu, 'BASE', Path(tmp)), patch.object(gpu, 'trainers', return_value=[live]), patch.object(gpu, 'status', return_value={'step': 1234}), patch.object(gpu, 'active_runtime', return_value={}), patch.object(gpu, 'graceful_stop') as stop, patch.object(gpu, 'launch') as launch, patch.object(gpu, 'receipt') as receipt:
            gpu.apply(ROOT, 'a'*40)
            stop.assert_not_called(); launch.assert_not_called()
            self.assertEqual(receipt.call_args.args[0], 'BOUND_IDENTICAL_NO_RESTART')
    def test_resumed_failed_trial_enters_rollback_handler(self):
        manifest = lib.validate_release(ROOT)
        live = {'pid': 123, 'source': '/existing/agillm44.py', 'source_sha256': manifest['source_sha256']}
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); tx = base / 'transactions' / ('a'*40) / 'transaction.json'
            lib.atomic_json(tx, {'phase': 'STARTED', 'checkpoint_step': 100})
            with patch.object(gpu, 'BASE', base), patch.object(gpu, 'trainers', return_value=[live]), patch.object(gpu, 'active_runtime', return_value={}), patch.object(gpu, 'wait_health', side_effect=ValueError('unhealthy')), patch.object(gpu, 'rollback_or_raise') as rollback:
                gpu.apply(ROOT, 'a'*40); rollback.assert_called_once()
    def test_invalid_commit_rejected_before_deployment(self):
        with self.assertRaises(ValueError): gpu.apply(ROOT, 'main; echo invalid')
    def test_three_advances_required_for_health(self):
        item = {'pid': 123}
        with patch.object(gpu, 'same_process', return_value=True), patch.object(gpu, 'status', side_effect=[{'step': 10}, {'step': 11}, {'step': 11}, {'step': 12}, {'step': 13}]), patch.object(gpu.time, 'sleep'):
            self.assertEqual(gpu.wait_health(item, 10)['step'], 13)

class CISelectionContracts(unittest.TestCase):
    def run_item(self, **kw):
        row = {'id': 1, 'head_sha': 'a'*40, 'head_branch': 'main', 'event': 'push', 'path': '.github/workflows/trainer-ci.yml', 'head_repository': {'full_name': 'owner/repo'}, 'status': 'completed', 'conclusion': 'success'}
        row.update(kw); return row
    def select(self, runs):
        with patch.object(pull, 'github', return_value={'workflow_runs': runs}): return pull.qualified_run('owner/repo', 'a'*40, 'trainer-ci.yml')
    def test_successful_main_push_accepted(self): self.assertIsNotNone(self.select([self.run_item()]))
    def test_pr_fork_failure_and_wrong_workflow_rejected(self):
        for change in ({'event': 'pull_request'}, {'head_repository': {'full_name': 'fork/repo'}}, {'conclusion': 'failure'}, {'head_sha': 'b'*40}, {'path': '.github/workflows/other.yml'}):
            with self.subTest(change=change): self.assertIsNone(self.select([self.run_item(**change)]))
    def test_new_failed_attempt_overrides_old_success(self):
        self.assertIsNone(self.select([self.run_item(run_attempt=1), self.run_item(run_attempt=2, conclusion='failure')]))

if __name__ == '__main__': unittest.main()
