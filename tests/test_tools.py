"""Unit tests for elderly_recsys_tools (synthetic data only, no D_lst)."""
import json
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import elderly_recsys_tools as erst


def act_context(times, places):
    """Activity context in the workbook layout (missing labels are NaN)."""
    nan = float("nan")
    times, places = (times + [nan] * 3)[:3], (places + [nan] * 3)[:3]
    return {"qID": "Q", **{f"act_C_T{i}": t for i, t in enumerate(times, 1)},
            **{f"act_C_P{i}": p for i, p in enumerate(places, 1)}}


CONTEXTS = {
    "A": act_context(["dopoldne"], ["doma"]),
    "B": act_context(["dopoldne"], ["v naravi"]),
    "ANYTIME": act_context(["kadarkoli"], ["doma"]),
    "ANYWHERE": act_context(["dopoldne"], ["kjerkoli"]),
}


class ItemIdentityTests(unittest.TestCase):
    def test_combinations_with_same_first_member_remain_distinct(self):
        combinations = [("A",), ("A", "B"), ("A", "C"), ("A", "B", "C")]
        self.assertEqual(len({erst.normalize_item(x) for x in combinations}), 4)

    def test_order_is_preserved(self):
        self.assertEqual(erst.normalize_item(["B", "A"]), ("B", "A"))
        self.assertNotEqual(erst.normalize_item(("A", "B")), erst.normalize_item(("B", "A")))

    def test_serialize_round_trip(self):
        for item in [("sprehod",), ("sprehod", "branje"), ("čaj", "A", "B")]:
            self.assertEqual(erst.normalize_item(json.loads(erst.serialize_item(item))), item)

    def test_empty_or_nested_items_are_rejected(self):
        for item in [(), [], "", (" ",), (("A", "B"),)]:
            with self.assertRaises(ValueError):
                erst.normalize_item(item)
        with self.assertRaises(TypeError):
            erst.normalize_item(5)


class ContextTests(unittest.TestCase):
    CTX = {"C_T": "dopoldne", "C_P": "doma"}

    def feasible(self, item, ctx=None):
        return erst.item_context_feasible(item, ctx or self.CTX, CONTEXTS)

    def test_incompatible_member_rejects_whole_combination(self):
        self.assertTrue(self.feasible(("A",)))
        self.assertFalse(self.feasible(("B",)))
        self.assertFalse(self.feasible(("A", "B")))
        self.assertFalse(self.feasible(("A", "unknown")))

    def test_time_and_place_must_both_match(self):
        self.assertFalse(self.feasible(("A",), {"C_T": "zvečer", "C_P": "doma"}))
        self.assertFalse(self.feasible(("A",), {"C_T": "dopoldne", "C_P": "v naravi"}))

    def test_kadarkoli_and_kjerkoli_relax_the_constraint(self):
        self.assertTrue(self.feasible(("ANYTIME",), {"C_T": "zvečer", "C_P": "doma"}))
        self.assertTrue(self.feasible(("ANYWHERE",), {"C_T": "dopoldne", "C_P": "v mestu"}))
        self.assertTrue(self.feasible(("A",), {"C_T": "kadarkoli", "C_P": "doma"}))
        self.assertTrue(self.feasible(("A",), {"C_T": "dopoldne", "C_P": "kjerkoli"}))
        # The relaxed dimension does not relax the other one
        self.assertFalse(self.feasible(("ANYTIME",), {"C_T": "zvečer", "C_P": "v mestu"}))
        self.assertFalse(self.feasible(("ANYWHERE",), {"C_T": "zvečer", "C_P": "v mestu"}))

    def test_context_pool_ignores_missing_labels(self):
        times, places = erst.build_context_pool(CONTEXTS)
        self.assertEqual(times, ["dopoldne", "kadarkoli"])
        self.assertEqual(places, ["doma", "kjerkoli", "v naravi"])


class GroupTests(unittest.TestCase):
    GROUPS = {"A": 0, "B": 1, "C": 0}

    def test_every_member_must_be_in_the_group(self):
        self.assertTrue(erst.item_in_group(("A", "C"), 0, self.GROUPS))
        self.assertFalse(erst.item_in_group(("A", "B"), 0, self.GROUPS))
        self.assertFalse(erst.item_in_group(("A", "B"), 1, self.GROUPS))

    def test_unknown_activity_is_in_no_group(self):
        self.assertFalse(erst.item_in_group(("unknown",), 0, self.GROUPS))
        self.assertFalse(erst.item_in_group(("A", "unknown"), 0, self.GROUPS))


class RankingTests(unittest.TestCase):
    def test_first_k_keeps_order_and_limit(self):
        ranked = [(1, (a,), s) for a, s in zip("ABCD", (4, 3, 2, 1))]
        self.assertEqual(erst.first_k(ranked, 2, lambda item: item != ("A",)), ranked[1:3])
        self.assertEqual(erst.first_k(ranked, 2, lambda item: False), [])


class EvaluateTopkTests(unittest.TestCase):
    def test_perfect_ranking(self):
        D = [(1, ("A",), 5.0), (1, ("B",), 4.0), (1, ("C",), 1.0)]
        m = erst.evaluate_topk(D, [(1, ("A",), 0.9), (1, ("B",), 0.8)], k=2, top_n=2)
        self.assertEqual(m, {"P": 1.0, "R": 1.0, "F1": 1.0, "F1_ties": 1.0, "n_users": 1})

    def test_user_without_recommendations_stays_in_the_average(self):
        D = [(1, ("A",), 5.0), (2, ("A",), 5.0)]
        m = erst.evaluate_topk(D, [(1, ("A",), 0.9)], k=1, top_n=1)
        self.assertEqual(m, {"P": 0.5, "R": 0.5, "F1": 0.5, "F1_ties": 0.5, "n_users": 2})
        empty = erst.evaluate_topk(D, [], k=1, top_n=1)
        self.assertEqual(empty, {"P": 0.0, "R": 0.0, "F1": 0.0, "F1_ties": 0.0, "n_users": 2})

    def test_user_without_reference_is_dropped(self):
        D = [(1, ("A",), 5.0), (2, ("B",), 5.0)]
        recs = [(1, ("A",), 0.9), (2, ("B",), 0.9)]
        m = erst.evaluate_topk(D, recs, k=1, top_n=1, gt_filter=lambda item: item == ("A",))
        self.assertEqual((m["F1"], m["n_users"]), (1.0, 1))
        none = erst.evaluate_topk(D, recs, k=1, top_n=1, gt_filter=lambda item: False)
        self.assertEqual((none["F1"], none["n_users"]), (0.0, 0))

    def test_ties_at_the_cutoff(self):
        # B and C are tied at the cut-off; the strict reference takes B (alphabetical)
        D = [(1, ("A",), 5.0), (1, ("B",), 3.0), (1, ("C",), 3.0), (1, ("D",), 1.0)]
        m = erst.evaluate_topk(D, [(1, ("A",), 0.9), (1, ("C",), 0.8)], k=2, top_n=2)
        self.assertEqual((m["F1"], m["F1_ties"]), (0.5, 1.0))
        m = erst.evaluate_topk(D, [(1, ("A",), 0.9), (1, ("D",), 0.8)], k=2, top_n=2)
        self.assertEqual((m["F1"], m["F1_ties"]), (0.5, 0.5))

    def test_items_are_compared_as_whole_tuples(self):
        m = erst.evaluate_topk([(1, ("A", "B"), 5.0)], [(1, ("A", "C"), 0.9)], k=1, top_n=1)
        self.assertEqual(m["F1"], 0.0)

    def test_short_list_has_precision_and_recall_of_its_own_length(self):
        D = [(1, (a,), r) for a, r in zip("ABCD", (4.0, 3.0, 2.0, 1.0))]
        m = erst.evaluate_topk(D, [(1, ("A",), 0.9)], k=5, top_n=4)
        self.assertEqual((m["P"], m["R"]), (1.0, 0.25))


class EvaluateGroupsTests(unittest.TestCase):
    def test_average_is_weighted_by_users(self):
        # Group 0: three users, all wrong; group 1: one user, right
        groups = {"A": 0, "B": 0, "C": 1}
        D = [(u, (a,), r) for u in (1, 2, 3) for a, r in (("A", 5.0), ("B", 1.0))] + [(1, ("C",), 2.0)]
        recs = {0: [(u, ("B",), 0.9) for u in (1, 2, 3)], 1: [(1, ("C",), 0.5)]}
        df, avg = erst.evaluate_groups(D, recs, groups, k=1, top_n=1, n_test_users=3, layer_name="M")
        self.assertEqual(list(df["NumUsers"]), [3, 1])
        self.assertEqual(list(df["F1"]), [0.0, 1.0])
        self.assertEqual(avg["F1"], 0.25)
        self.assertEqual(list(df["Coverage"]), [1.0, 0.333])


class ScoreTests(unittest.TestCase):
    def test_score_and_compatibility_penalty(self):
        answers = pd.DataFrame({"A": [1.0], "B": [0.5]}, index=[7])
        scores = pd.DataFrame({"A": [2.0], "B": [4.0]}, index=[7])
        compat = pd.DataFrame([[0.0, 0.0], [0.0, 0.0]], index=["A", "B"], columns=["A", "B"])
        self.assertEqual(erst.get_score_estimation(7, ("A",), answers, scores, compat, "score"), 2.0)
        self.assertEqual(erst.get_score_estimation(7, ("A", "B"), answers, scores, compat, "score"), 3.0)

    def test_unknown_method_is_rejected(self):
        with self.assertRaises(ValueError):
            erst.get_score_estimation(7, ("A",), None, None, None, "unknown")


if __name__ == "__main__":
    unittest.main()
