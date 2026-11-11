import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from analysis.summarize import episode,summarize
from analysis.paired_uncertainty import compute
from scripts.verify_source import verify

class ReleaseChecks(unittest.TestCase):
    def test_manifest_integrity(self):
        self.assertEqual(verify()['failures'],[])

    def test_task_routes_and_ids(self):
        tasks=json.loads((ROOT/'configs/paper_tasks.json').read_text())
        self.assertEqual(len(tasks),36)
        self.assertEqual(len({x['task_id'] for x in tasks}),36)
        self.assertEqual(len({x['map'] for x in tasks}),5)
        for t in tasks:
            data=json.loads((ROOT/t['task_file']).read_text())['tasks'][t['task_index']]
            self.assertEqual(data['task_id'],t['task_id'])

    def test_native_examples_preserve_recorded_metrics(self):
        for item in json.loads((ROOT/'examples/manifest.json').read_text()):
            p=ROOT/'examples'/item['file'];raw=json.loads(p.read_text());r=episode(p)
            self.assertEqual(r['success'],raw['benchmark_metrics']['success_indicator'])
            self.assertEqual(r['safe_success'],raw['benchmark_metrics']['safe_success_indicator'])
            self.assertAlmostEqual(r['spl'],raw['spl'])
            self.assertEqual(r['collisions'],raw['collision_count'])

    def test_safety_penalty_and_exposure_denominator(self):
        paths=list((ROOT/'examples').glob('*/episode.json'))
        rows=[episode(p) for p in paths]
        for row in rows:
            self.assertEqual(row['safety_score'],-(row['collisions']+row['hazards']+10*row['violations']))
        # Group the two real records only for testing ratio-of-totals arithmetic.
        merged=[dict(r,mode='test') for r in rows];out=summarize(merged)[0]
        self.assertAlmostEqual(out['collisions_per_100m_traveled'],100*sum(r['collisions'] for r in rows)/sum(r['distance_m'] for r in rows))

    def test_paired_analysis_rejects_unmatched_examples(self):
        rows=[episode(p) for p in (ROOT/'examples').glob('*/episode.json')]
        with self.assertRaises(ValueError):compute(rows,20)
        base=rows[0]
        paired=[dict(base,task_id=0,mode='static'),dict(base,task_id=0,mode='realtime',collisions=base['collisions']+2)]
        result=compute(paired,20)
        self.assertEqual(result['realtime_minus_static']['collisions']['mean'],2)

    def test_analysis_rejects_mixed_runtime_protocols(self):
        row=episode(next((ROOT/'examples').glob('*/episode.json')))
        mixed=[row,dict(row,runtime_protocol='different-runtime',task_id=999)]
        with self.assertRaises(ValueError):summarize(mixed)
        with self.assertRaises(ValueError):compute(mixed,20)

    def test_paper_plans_preserve_model_and_counts(self):
        for study,n in [('rq1',108),('rq2',72),('rq3',108)]:
            completed=subprocess.run([sys.executable,str(ROOT/'scripts/run_paper.py'),'--model','Gemini','--study',study],capture_output=True,text=True,check=True)
            plan=json.loads(completed.stdout)
            self.assertFalse(plan['starts_rollouts'])
            self.assertEqual(plan['episodes'],n)
            self.assertEqual(plan['environment']['SIMWORLD_CAMPAIGN_MODEL'],'google/gemini-3.8-flash')
            self.assertEqual(plan['environment']['SIMWORLD_CAMPAIGN_API_MODE'],'chat_completions')

if __name__=='__main__':unittest.main()
