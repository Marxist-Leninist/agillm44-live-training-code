"""Real-filesystem regression tests. No trainer imports, GPU use, or networking."""
from __future__ import annotations
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get('ATOMIC_JSON_SOURCE', str(ROOT / 'deploy/release_lib.py')))
spec = importlib.util.spec_from_file_location('atomic_json_target', SOURCE)
lib = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lib)

@unittest.skipUnless(os.name == 'posix' and hasattr(os, 'O_DIRECTORY'), 'POSIX implementation')
class AtomicJsonTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path = self.root / 'status.json'
        self.original = b'{"step": 10}\n'
        self.path.write_bytes(self.original)

    def assert_no_temporary_files(self):
        self.assertEqual(list(self.root.glob(self.path.name + '.tmp.*')), [])

    def test_format_roundtrip(self):
        value = {'step': 11, 'nested': [1, True, None, {'label': 'SAT-var \u03b1'}]}
        lib.atomic_json(self.path, value)
        self.assertEqual(self.path.read_text(), json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
        self.assert_no_temporary_files()

    def test_creates_parent_directories(self):
        target = self.root / 'new' / 'nested' / 'status.json'
        lib.atomic_json(target, {'step': 11})
        self.assertEqual(json.loads(target.read_text()), {'step': 11})

    def test_replaces_instead_of_appending(self):
        self.path.write_text('X' * 100_000)
        lib.atomic_json(self.path, {'step': 11})
        self.assertEqual(json.loads(self.path.read_text()), {'step': 11})
        self.assertLess(self.path.stat().st_size, 100)

    def test_published_mode_is_private(self):
        self.path.chmod(0o666)
        lib.atomic_json(self.path, {'step': 11})
        self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o600)

    def test_invalid_json_leaves_old_file_and_no_temporary(self):
        for value in ({'loss': float('nan')}, {'loss': float('inf')}, {'bad': object()}):
            with self.subTest(value=value):
                with self.assertRaises((ValueError, TypeError)):
                    lib.atomic_json(self.path, value)
                self.assertEqual(self.path.read_bytes(), self.original)
                self.assert_no_temporary_files()

    def test_replace_failure_preserves_old_file_and_cleans_up(self):
        with patch.object(lib.os, 'replace', side_effect=OSError('injected replace failure')):
            with self.assertRaisesRegex(OSError, 'injected replace'):
                lib.atomic_json(self.path, {'step': 11})
        self.assertEqual(self.path.read_bytes(), self.original)
        self.assert_no_temporary_files()

    def test_file_fsync_failure_preserves_old_file_and_cleans_up(self):
        with patch.object(lib.os, 'fsync', side_effect=OSError('injected file fsync')):
            with self.assertRaisesRegex(OSError, 'file fsync'):
                lib.atomic_json(self.path, {'step': 11})
        self.assertEqual(self.path.read_bytes(), self.original)
        self.assert_no_temporary_files()

    def test_directory_fsync_failure_is_reported_after_publication(self):
        fsync = lib.os.fsync
        def reject_directory(fd):
            if stat.S_ISDIR(os.fstat(fd).st_mode):
                raise OSError('injected directory fsync')
            return fsync(fd)
        with patch.object(lib.os, 'fsync', side_effect=reject_directory):
            with self.assertRaisesRegex(OSError, 'directory fsync'):
                lib.atomic_json(self.path, {'step': 11})
        self.assertEqual(json.loads(self.path.read_text()), {'step': 11})
        self.assert_no_temporary_files()

    def test_flush_and_fsync_order(self):
        fsync, replace = lib.os.fsync, lib.os.replace
        events = []
        def record_fsync(fd):
            is_dir = stat.S_ISDIR(os.fstat(fd).st_mode)
            events.append('directory_fsync' if is_dir else 'file_fsync')
            if not is_dir:
                self.assertGreater(os.fstat(fd).st_size, 0)
            return fsync(fd)
        def record_replace(src, dst):
            events.append('replace')
            self.assertEqual(json.loads(Path(src).read_text()), {'step': 11})
            return replace(src, dst)
        with patch.object(lib.os, 'fsync', side_effect=record_fsync), patch.object(lib.os, 'replace', side_effect=record_replace):
            lib.atomic_json(self.path, {'step': 11})
        self.assertEqual(events, ['file_fsync', 'replace', 'directory_fsync'])

    def test_same_process_writers_do_not_share_temporary_file(self):
        replace = lib.os.replace
        barrier = threading.Barrier(2, timeout=10)
        paths = []
        def simultaneous_replace(src, dst):
            paths.append(str(src))
            barrier.wait()
            return replace(src, dst)
        values = [{'writer': i, 'payload': str(i) * (5000 + 1000 * i)} for i in range(2)]
        with patch.object(lib.os, 'replace', side_effect=simultaneous_replace):
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(lib.atomic_json, self.path, value) for value in values]
                outcomes = []
                for future in futures:
                    try:
                        future.result(timeout=15)
                        outcomes.append('ok')
                    except Exception as exc:
                        outcomes.append(type(exc).__name__)
        self.assertEqual(outcomes, ['ok', 'ok'])
        self.assertEqual(len(set(paths)), 2)
        self.assertIn(json.loads(self.path.read_text()), values)
        self.assert_no_temporary_files()

    def test_legacy_predictable_symlink_is_not_opened(self):
        victim = self.root / 'unrelated.txt'
        victim.write_bytes(b'preserve this unrelated file')
        legacy = self.path.with_name(self.path.name + '.tmp.' + str(os.getpid()))
        legacy.symlink_to(victim)
        lib.atomic_json(self.path, {'step': 11})
        self.assertEqual(victim.read_bytes(), b'preserve this unrelated file')
        self.assertTrue(legacy.is_symlink())
        legacy.unlink()
        self.assert_no_temporary_files()

    def test_interrupt_during_serialization_cleans_up(self):
        with patch.object(lib.json, 'dump', side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                lib.atomic_json(self.path, {'step': 11})
        self.assertEqual(self.path.read_bytes(), self.original)
        self.assert_no_temporary_files()

    def test_reader_never_observes_partial_json(self):
        stop = threading.Event()
        errors = []
        count = [0]
        started = threading.Event()
        def reader():
            while not stop.is_set():
                try:
                    row = json.loads(self.path.read_text())
                    self.assertIn('step', row)
                    count[0] += 1
                    started.set()
                except Exception as exc:
                    errors.append(repr(exc))
                    started.set()
                    return
        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        try:
            self.assertTrue(started.wait(timeout=5))
            for i in range(50):
                lib.atomic_json(self.path, {'step': i, 'payload': 'x' * (2000 + i)})
        finally:
            stop.set()
            thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertGreater(count[0], 0)
        self.assert_no_temporary_files()

    @unittest.skipUnless(hasattr(lib, 'tempfile'), 'new unique-temporary implementation only')
    def test_fdopen_failure_closes_descriptor_and_cleans_up(self):
        mkstemp = lib.tempfile.mkstemp
        descriptors = []
        def record_temporary(*args, **kwargs):
            result = mkstemp(*args, **kwargs)
            descriptors.append(result[0])
            return result
        with patch.object(lib.tempfile, 'mkstemp', side_effect=record_temporary), patch.object(lib.os, 'fdopen', side_effect=OSError('injected fdopen failure')):
            with self.assertRaisesRegex(OSError, 'fdopen failure'):
                lib.atomic_json(self.path, {'step': 11})
        self.assertEqual(len(descriptors), 1)
        with self.assertRaises(OSError):
            os.fstat(descriptors[0])
        self.assertEqual(self.path.read_bytes(), self.original)
        self.assert_no_temporary_files()

    @unittest.skipUnless(hasattr(lib, 'tempfile'), 'new unique-temporary implementation only')
    def test_permission_failure_closes_descriptor_and_cleans_up(self):
        mkstemp = lib.tempfile.mkstemp
        descriptors = []
        def record_temporary(*args, **kwargs):
            result = mkstemp(*args, **kwargs)
            descriptors.append(result[0])
            return result
        with patch.object(lib.tempfile, 'mkstemp', side_effect=record_temporary), patch.object(lib.os, 'fchmod', side_effect=OSError('injected mode failure')):
            with self.assertRaisesRegex(OSError, 'mode failure'):
                lib.atomic_json(self.path, {'step': 11})
        self.assertEqual(len(descriptors), 1)
        with self.assertRaises(OSError):
            os.fstat(descriptors[0])
        self.assertEqual(self.path.read_bytes(), self.original)
        self.assert_no_temporary_files()

if __name__ == '__main__':
    unittest.main()
