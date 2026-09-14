"""CPU-only tests for fresh Vast.ai SSH endpoint resolution."""
from __future__ import annotations
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'deploy'))
import pull_release as pull


class VastEndpointContracts(unittest.TestCase):
    def test_resolves_current_mapped_ssh_endpoint(self):
        payload = json.dumps([{
            'id': 51049010,
            'actual_status': 'running',
            'cur_state': 'running',
            'public_ipaddr': '107.206.71.138',
            'ssh_host': 'ssh8.vast.ai',
            'ssh_port': 19010,
            'ports': {'22/tcp': [{'HostIp': '0.0.0.0', 'HostPort': '46963'}]},
        }]).encode()
        with patch.object(pull, 'run', return_value=payload) as runner:
            self.assertEqual(pull.resolve_destination({'vast_instance_id': 51049010}), ('107.206.71.138', '46963'))
            self.assertEqual(runner.call_args.args[0], ['vastai', 'show', 'instances', '--raw'])

    def test_falls_back_to_vast_ssh_host_when_port_map_missing(self):
        payload = json.dumps([{
            'id': 9,
            'actual_status': 'running',
            'cur_state': 'running',
            'public_ipaddr': '',
            'ssh_host': 'ssh9.vast.ai',
            'ssh_port': 19011,
            'ports': {},
        }]).encode()
        with patch.object(pull, 'run', return_value=payload):
            self.assertEqual(pull.resolve_destination({'vast_instance_id': 9}), ('ssh9.vast.ai', '19011'))

    def test_static_destination_remains_backward_compatible(self):
        self.assertEqual(pull.resolve_destination({'host': 'example.test', 'port': 2200}), ('example.test', '2200'))

    def test_stopped_instance_is_rejected(self):
        payload = json.dumps([{'id': 7, 'actual_status': 'exited', 'cur_state': 'stopped'}]).encode()
        with patch.object(pull, 'run', return_value=payload):
            with self.assertRaisesRegex(ValueError, 'not running'):
                pull.resolve_destination({'vast_instance_id': 7})

    def test_wrong_instance_is_rejected(self):
        payload = json.dumps([{'id': 8, 'actual_status': 'running', 'cur_state': 'running'}]).encode()
        with patch.object(pull, 'run', return_value=payload):
            with self.assertRaisesRegex(ValueError, 'not found'):
                pull.resolve_destination({'vast_instance_id': 7})


if __name__ == '__main__':
    unittest.main()
