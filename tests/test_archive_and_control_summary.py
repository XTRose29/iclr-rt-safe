import copy
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from analysis.summarize import episode, summarize
from analysis.paired_uncertainty import compute
from evaluation import run_qwen3vl8b_all_maps as runner

ROOT = Path(__file__).resolve().parents[1]


class ArchiveAndSummaryChecks(unittest.TestCase):
    def test_archive_provenance_without_git_is_stable_and_tracks_edits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'data').mkdir()
            (root/'base').mkdir()
            (root/'config.yaml').write_text('setting: 1\n')
            (root/'data/ue_assets.json').write_text('{}')
            source = root/'base/agent.py'
            source.write_text('value = 1\n')
            args = SimpleNamespace(ue_launcher=str(root/'runtime/SimWorld.sh'),
                                   qwen_model_path=str(root/'weights'))
            with patch.object(runner.subprocess, 'run', side_effect=AssertionError('Git must not be required')):
                first = runner.build_source_provenance([], args, repo_root=root)
                runner.require_clean_source(first)
                state = first['repository']
                self.assertIsNone(state['commit'])
                self.assertIsNone(state['tracked_worktree_clean'])
                self.assertIn('base/agent.py', state['source_file_sha256'])
                (root/'results').mkdir()
                (root/'results/episode.json').write_text('{}')
                self.assertEqual(state, runner.git_source_state(root))
                source.write_text('value = 2\n')
                self.assertNotEqual(state['source_tree_sha256'], runner.git_source_state(root)['source_tree_sha256'])

    def test_dirty_checkout_is_still_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'.git').mkdir()
            replies = [SimpleNamespace(stdout=' M base/agent.py\n'), SimpleNamespace(stdout='abcdef\n')]
            with patch.object(runner.subprocess, 'run', side_effect=replies):
                state = runner.git_source_state(root)
            with self.assertRaises(RuntimeError):
                runner.require_clean_source({'repository': state})
        with self.assertRaises(RuntimeError):
            runner.require_clean_source({'repository': {'source_type': 'archive'}})

    def make_records(self, root):
        original = json.loads((ROOT/'examples/native_task0/episode.json').read_text())
        records = []
        for effort in ('low', 'medium', 'xhigh'):
            config = root/f'{effort}.json'
            runner.write_agent_config(config, runner.parse_args([
                '--model', 'Qwen/Qwen3.8-27B-FP8', '--enable-thinking', '--reasoning-effort', effort]))
            llm = json.loads(config.read_text())['llm'][0]
            self.assertEqual(llm['reasoning_effort'], effort)
            for delay in (0, 3, 5):
                data = copy.deepcopy(original)
                data.update(model=llm['model'], control_mode='llm', reasoning_effort=llm['reasoning_effort'],
                            llm_extra_body=llm['extra_body'], realtime_thinking=True)
                data['experiment_controls'] = dict(fixed_thinking_seconds=delay, prompt_timing_mode='neutral')
                # Also accept records written before top-level effort was persisted.
                if delay == 3:
                    data['reasoning_effort'] = None
                path = root/f'{effort}_{delay}.json'
                path.write_text(json.dumps(data))
                records.append(path)
        return records

    def test_nine_conditions_survive_cli_and_csv_and_duplicates_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = self.make_records(root)
            command = [sys.executable, str(ROOT/'analysis/summarize.py'), '--episodes',
                       *map(str, records), '--output', str(root/'output')]
            completed = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            with (root/'output/summary.csv').open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 9)
            self.assertEqual({(r['effort'], float(r['fixed_thinking_seconds'])) for r in rows},
                             {(e, d) for e in ('low', 'medium', 'xhigh') for d in (0, 3, 5)})
            self.assertTrue(all(r['episodes'] == '1' for r in rows))
            duplicate = command[:]
            duplicate.insert(duplicate.index('--output'), str(records[0]))
            completed = subprocess.run(duplicate, capture_output=True, text=True)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn('Duplicate', completed.stderr)

    def test_measured_delay_remains_distinct_from_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = self.make_records(root)
            row = episode(records[0])
            rows = [row, dict(row, fixed_thinking_seconds=None)]
            self.assertEqual(len(summarize(rows)), 2)
            paired = [dict(row, mode='static'), dict(row, mode='realtime', fixed_thinking_seconds=3)]
            with self.assertRaisesRegex(ValueError, 'fixed_thinking_seconds'):
                compute(paired, 10)

    def test_conflicting_effort_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self.make_records(root)[0]
            data = json.loads(path.read_text())
            data['reasoning_effort'] = 'xhigh'
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'Conflicting'):
                episode(path)
