from pathlib import Path
import random
import json
from tempfile import TemporaryDirectory
import unittest

from evolution import Candidate, front, hypervolume, ox, rank_crowd, solve
from scheduling import check, decode, read_case, Case


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.case = read_case(Path(__file__).resolve().parents[1]/"data"/"balanced.json")

    def test_hand_schedule(self):
        c=Case("tiny",("A",),(((3,5),),),({"release":0,"due":7,"weight":3},),
               ({"job":0,"family":0},{"job":0,"family":1}),((),(0,)),((1,),()),
               ((0,),(0,)),({0:3},{0:2}),((0,2),(4,0)),(20,20,20))
        f,plan=decode(c,(1,0),(0,0))
        self.assertEqual(plan,[(0,0,0,3),(0,5,7,9)])
        self.assertEqual(check(c,plan), (9,6,2))
        self.assertEqual(f,(9,6,2))
        with self.assertRaises(ValueError):
            check(c,[(0,0,0,3),(0,3,5,7)])

    def test_feasible_and_checker_detects_corruption(self):
        rng=random.Random(62)
        n=len(self.case.ops)
        for _ in range(100):
            f,rows=decode(self.case,rng.sample(range(n),n),[rng.choice(rs) for rs in self.case.eligible])
            self.assertEqual(f,check(self.case,rows))
        r,s,p,e=rows[0]
        rows[0]=(r,s,p,e+1)
        with self.assertRaises(ValueError):
            check(self.case,rows)

    def test_invalid_chromosome_is_rejected(self):
        n = len(self.case.ops)
        order = list(range(n))
        machines = [r[0] for r in self.case.eligible]
        with self.assertRaisesRegex(ValueError, "перестановкой"):
            decode(self.case, [0]*n, machines)
        with self.assertRaisesRegex(ValueError, "допустимый ресурс"):
            decode(self.case, order, [-1]+machines[1:])

    def test_fractional_operation_index_is_rejected(self):
        source = Path(__file__).resolve().parents[1]/"data"/"balanced.json"
        data = json.loads(source.read_text())
        data["operations"][0]["family"] = 0.5
        with TemporaryDirectory() as tmp:
            p = Path(tmp)/"bad.json"
            p.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "операция O0"):
                read_case(p)

    def test_ox_preserves_permutation(self):
        rng=random.Random(5)
        for _ in range(100):
            a,b=(tuple(rng.sample(range(22),22)) for _ in range(2))
            self.assertEqual(sorted(ox(a,b,rng)),list(range(22)))

    def test_pareto_and_volume(self):
        cand=[Candidate((),(),x) for x in ((1,4,2),(4,1,2),(4,4,3),(1,4,2))]
        self.assertEqual([c.f for c in front(cand)],[(1,4,2),(4,1,2)])
        rank,_=rank_crowd(cand)
        self.assertEqual(list(rank),[0,0,1,0])
        self.assertAlmostEqual(hypervolume((c.f for c in front(cand)),(5,5,5)),21/125)

    def test_reproducible_identical_budget(self):
        for local in (False,True):
            x=solve(self.case,15,80,8,local)
            self.assertEqual(x,solve(self.case,15,80,8,local))
            self.assertEqual(x[2]["evaluations"],80)
            self.assertEqual(x[2]["valid"],80)
            self.assertEqual(x[2]["attempts"],18 if local else 0)
