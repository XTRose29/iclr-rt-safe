"""Offline check that the batch helper covers all loaded tasks."""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]


def method(path, class_name, name):
    tree = ast.parse(path.read_bytes())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)


def without_docstring(node):
    if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
        node.body = node.body[1:]
    return ast.dump(node, include_attributes=False)


class CleanupChecks(unittest.TestCase):
    def test_batch_covers_all_tasks_including_small_task_sets(self):
        # Execute the actual helper with simulator/model calls replaced by stubs.
        node = method(ROOT/'manager/world_manager.py', 'WorldManager', 'run_multiple_tasks')
        namespace = {'time': SimpleNamespace(sleep=lambda _: None)}
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<batch-helper>', 'exec'), namespace)
        for count in (1, 5):
            with self.subTest(task_count=count):
                calls = []
                fake = SimpleNamespace(
                    all_scenarios=[{'task_id': 100+i} for i in range(count)],
                    model_configs=[{'llm': {'model': 'a'}}, {'llm': {'model': 'b'}}],
                    logger=Mock(), _create_llm_from_config=lambda _: None,
                    cleanup=Mock(),
                )
                def run_task(index, difficulty):
                    fake.current_task_id = fake.all_scenarios[index]['task_id']
                    fake.agent = SimpleNamespace(success=True, step_num=1, decision_count=1)
                    calls.append((index, difficulty, fake.config['llm']['model']))
                fake.run_single_task = run_task
                results = namespace['run_multiple_tasks'](fake, repeats_per_task=2)
                expected = Counter({(i, d, m): 2 for i in range(count) for d in ('default', 'medium', 'easy') for m in ('a', 'b')})
                self.assertEqual(Counter(calls), expected)
                self.assertEqual(len(results), count*3*2*2)
                self.assertTrue(all(r['success'] for r in results))
                self.assertEqual(fake.cleanup.call_count, len(results))


if __name__ == '__main__':
    unittest.main()
