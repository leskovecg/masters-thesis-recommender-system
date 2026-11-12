import ast
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import recommendation_items as items


class ItemIdentityTests(unittest.TestCase):
    def test_combinations_with_same_first_member_remain_distinct(self):
        combinations = [("A",), ("A", "B"), ("A", "C"), ("A", "B", "C")]
        self.assertEqual(len({items.normalize_item(x) for x in combinations}), 4)
        factors = {items.normalize_item(k): i for i, k in enumerate(combinations)}
        self.assertEqual(factors[("A", "B")], 1)
        self.assertEqual(factors[("A", "C")], 2)

    def test_export_round_trip_and_reference_keep_same_item(self):
        item = ("sprehod", "branje")
        exported = items.serialize_item(item)
        recovered = items.normalize_item(json.loads(exported))
        reference = items.build_reference_triplets([(2, item, (), 4.2)])
        self.assertEqual(reference, [(2, recovered, 4.2)])

    def test_incompatible_second_action_rejects_whole_combination(self):
        allowed = {"A": "doma", "B": "zunaj"}
        check = lambda a, c, d: d.get(a) == c["place"]
        self.assertTrue(items.item_context_feasible(("A",), {"place": "doma"}, allowed, check))
        self.assertFalse(items.item_context_feasible(("A", "B"), {"place": "doma"}, allowed, check))
        self.assertFalse(items.item_context_feasible(("unknown",), {"place": "doma"}, allowed, check))

    def test_group_membership_checks_every_member(self):
        groups = {"A": 0, "B": 1, "C": 0}
        self.assertTrue(items.item_in_group(("A", "C"), 0, groups))
        self.assertFalse(items.item_in_group(("A", "B"), 0, groups))
        self.assertFalse(items.item_in_group(("unknown",), 0, groups))

    def test_empty_or_nested_items_are_rejected(self):
        for item in [(), [], "", (("A", "B"),)]:
            with self.assertRaises(ValueError):
                items.normalize_item(item)

    def test_v7_integration_keeps_items_and_factor_keys(self):
        """Execute the actual v7 expressions with lightweight model substitutes."""
        tree = ast.parse((ROOT / "src/runGeneratorOfD_v7.py").read_text(encoding="utf-8"))
        candidate_assignment = next(n for n in ast.walk(tree)
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "topN_norm" for t in n.targets))
        namespace = {"items": items, "topN": [(2, ("A", "B"), 4.2), (2, ("A", "C"), 4.1)]}
        exec(compile(ast.Module(body=[candidate_assignment], type_ignores=[]), "v7", "exec"), namespace)
        namespace.update(uID=2, top5_41=namespace["topN_norm"], recs_41=[])
        extend_call = next(n for n in ast.walk(tree)
            if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
            and isinstance(n.value.func, ast.Attribute)
            and isinstance(n.value.func.value, ast.Name)
            and n.value.func.value.id == "recs_41" and n.value.func.attr == "extend")
        exec(compile(ast.Module(body=[extend_call], type_ignores=[]), "v7", "exec"), namespace)
        self.assertEqual(namespace["recs_41"], [(2, ("A", "B"), 4.2), (2, ("A", "C"), 4.1)])
        factory = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "make_gt_filter_for_group")
        exec(compile(ast.Module(body=[factory], type_ignores=[]), "v7", "exec"), namespace)
        filt = namespace["make_gt_filter_for_group"]({"A": 0, "B": 1}, 0)
        self.assertFalse(filt(("A", "B")))


if __name__ == "__main__":
    unittest.main()
