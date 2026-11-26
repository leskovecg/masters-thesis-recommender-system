"""Unit tests for elderly_recsys_tools (synthetic data only, no D_lst)."""
import itertools
import json
import sys
import unittest
from pathlib import Path

import numpy as np
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

    def test_tie_aware_hits_are_capped_at_reference_size(self):
        # Three tied items, one reference item, three recommendations: F1_ties must not exceed 1
        D = [(1, (a,), 5.0) for a in "ABC"]
        recs = [(1, (a,), s) for a, s in zip("ABC", (0.9, 0.8, 0.7))]
        m = erst.evaluate_topk(D, recs, k=3, top_n=1)
        self.assertAlmostEqual(m["F1_ties"], 0.5)  # P = 1/3, R = 1
        self.assertLessEqual(m["F1_ties"], 1.0)

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
        self.assertEqual(list(df["Coverage"]), [1.0, 1 / 3])

    def test_average_uses_unrounded_group_metrics(self):
        # Group 0: F1 = 1/3 (three users); group 1: F1 = 0 (one user)
        groups = {"A": 0, "B": 0, "C": 1, "D": 1}
        D = [(u, (a,), r) for u in (1, 2, 3) for a, r in (("A", 5.0), ("B", 1.0))] + [(1, ("C",), 2.0), (1, ("D",), 1.0)]
        recs = {0: [(1, ("A",), 0.9), (2, ("B",), 0.9), (3, ("B",), 0.9)], 1: [(1, ("D",), 0.5)]}
        df, avg = erst.evaluate_groups(D, recs, groups, k=1, top_n=1, n_test_users=3, layer_name="M")
        self.assertEqual(avg["F1"], 0.25)
        # Mean over all returned recommendations: (5 + 1 + 1 + 1) / 4
        self.assertEqual(avg["AverageReferenceRating"], 2.0)


class ExpectedRandomTests(unittest.TestCase):
    D = [(1, (a,), r) for a, r in zip("ABCD", (4.0, 3.0, 2.0, 1.0))]

    def simulate(self, reference, candidates, k, top_n):
        """Exact expectation over all orders of the candidates of a single user."""
        runs = [erst.evaluate_topk(reference, [(1, item, -i) for i, (_, item, _) in enumerate(order)], k, top_n)
                for order in itertools.permutations(candidates)]
        return {m: sum(run[m] for run in runs) / len(runs) for m in ("P", "R", "F1", "F1_ties")}

    def check(self, reference, candidates, k, top_n):
        expected = erst.expected_random_topk(reference, candidates, k, top_n)
        for m, value in self.simulate(reference, candidates, k, top_n).items():
            self.assertAlmostEqual(expected[m], value, places=12, msg=m)
        return expected

    def test_candidates_equal_to_reference_pool(self):
        # E[F1] = m / n with m = 2, n = 4
        self.assertEqual(self.check(self.D, self.D, 2, 2)["F1"], 0.5)

    def test_candidates_are_a_subset_of_the_reference_pool(self):
        self.check(self.D, self.D[1:], 2, 2)
        self.check(self.D, self.D[:1], 2, 2)  # fewer candidates than k
        tied = [(1, (a,), r) for a, r in zip("ABCD", (4.0, 3.0, 3.0, 1.0))]
        self.check(tied, tied[1:], 2, 2)

    def test_tie_aware_cap_when_more_recommendations_than_reference(self):
        # k > top_n: the exact expectation of the capped hits matches all orderings
        tied = [(1, (a,), r) for a, r in zip("ABCD", (5.0, 5.0, 5.0, 1.0))]
        self.assertLessEqual(self.check(tied, tied, 3, 1)["F1_ties"], 1.0)

    def test_no_candidates_at_all(self):
        m = erst.expected_random_topk(self.D, [], 2, 2)
        self.assertEqual((m["F1"], m["n_users"], m["n_covered"]), (0.0, 1, 0))
        self.assertTrue(np.isnan(m["avg_rating"]))

    def test_short_list_returns_everything(self):
        self.assertEqual(self.check(self.D, self.D, 5, 5)["F1"], 1.0)

    def test_user_without_candidates_counts_as_zero(self):
        m = erst.expected_random_topk(self.D + [(2, ("A",), 1.0)], self.D, 4, 4)
        self.assertEqual((m["F1"], m["n_users"], m["n_covered"]), (0.5, 2, 1))

    def test_expected_rating_is_weighted_by_returned_items(self):
        D = self.D + [(2, ("A",), 5.0)]
        # User 1 returns 2 of 4 items (mean 2.5), user 2 returns 1 item (mean 5)
        self.assertAlmostEqual(erst.expected_random_topk(D, D, 2, 2)["avg_rating"], (2 * 2.5 + 5.0) / 3)


class ContextBreakdownTests(unittest.TestCase):
    def test_users_are_split_by_feasible_items(self):
        D = [(1, (a,), r) for a, r in zip("ABC", (3.0, 2.0, 1.0))] + [(2, ("A",), 1.0)]
        recs = [(1, ("C",), 0.9), (2, ("A",), 0.9)]
        df = erst.context_breakdown(D, {"SVD": recs}, {1, 2, 3, 4}, {1, 2, 3}, k=1, top_n=1)
        self.assertEqual(list(df["NumUsers"]), [1, 1, 1, 1, 2])
        self.assertEqual(list(df["F1_SVD"])[2:], [1.0, 0.0, 0.5])
        self.assertTrue(df["F1_SVD"][:2].isna().all())
        self.assertAlmostEqual(df["F1_Random"][3], 1 / 3)


class ScoreTests(unittest.TestCase):
    def test_score_and_compatibility_penalty(self):
        answers = pd.DataFrame({"A": [1.0], "B": [0.5]}, index=[7])
        scores = pd.DataFrame({"A": [2.0], "B": [4.0]}, index=[7])
        compat = pd.DataFrame([[0.0, 0.0], [0.0, 0.0]], index=["A", "B"], columns=["A", "B"])
        self.assertEqual(erst.get_score_estimation(7, ("A",), answers, scores, compat, "score"), 2.0)
        self.assertEqual(erst.get_score_estimation(7, ("A", "B"), answers, scores, compat, "score"), 3.0)

    def test_numpy_integer_answer_is_a_number(self):
        answers = pd.DataFrame({"Q": np.array([1, 0], dtype=np.int64)}, index=[7, 8])
        scores = erst.get_actID_score_df([7, 8], ["A"], {"A": "Q"}, {7: 2.5, 8: np.int64(3)}, answers)
        self.assertEqual((scores.at[7, "A"], scores.at[8, "A"]), (2.5, 0))

    def test_missing_answer_propagates_nan_and_text_gives_zero(self):
        answers = pd.DataFrame({"Q": [np.nan, " "]}, index=[7, 8], dtype=object)
        scores = erst.get_actID_score_df([7, 8], ["A"], {"A": "Q"}, {7: 2.0, 8: 2.0}, answers)
        self.assertTrue(np.isnan(scores.at[7, "A"]))
        self.assertEqual(scores.at[8, "A"], 0)

    def test_scaled_answers_treat_missing_and_text_as_zero(self):
        answers = pd.DataFrame({"Q1": [0.0, 2.0, np.nan], "Q2": [1, " ", 3], "X": ["a", "b", "c"]}, index=[1, 2, 3])
        scaled = erst.get_uID_answers_df(answers, {"g": ["Q1", "Q2"]}, ["g"])
        self.assertEqual(list(scaled.columns), ["Q1", "Q2"])
        self.assertEqual(scaled.index.names, ["uID"])
        self.assertEqual(scaled["Q1"].tolist(), [0.0, 1.0, 0.0])
        self.assertEqual(scaled["Q2"].tolist(), [1 / 3, 0.0, 1.0])

    def test_unknown_method_is_rejected(self):
        with self.assertRaises(ValueError):
            erst.get_score_estimation(7, ("A",), None, None, None, "unknown")


if __name__ == "__main__":
    unittest.main()
