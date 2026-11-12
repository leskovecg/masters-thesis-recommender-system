"""Preserve complete activity items from prediction through export and reference.

An item is a nonempty tuple of activity IDs. Tuple order is retained, because
the existing Surprise model uses that exact tuple as its key.
"""
import json


def normalize_item(item):
    if isinstance(item, str):
        item = (item,)
    elif isinstance(item, (tuple, list)):
        item = tuple(item)
    else:
        raise TypeError("Expected an activity ID or a tuple/list of activity IDs")
    if not item or any(not isinstance(a, str) or not a.strip() for a in item):
        raise ValueError("An item must contain nonempty string activity IDs")
    return item


def serialize_item(item):
    """Lossless JSON representation for Excel exports (not a model key)."""
    return json.dumps(normalize_item(item), ensure_ascii=False)


def build_reference_triplets(D_lst):
    """Retain every combination; this does NOT perform a held-out split."""
    return [(uid, normalize_item(item), float(rating))
            for uid, item, context, rating in D_lst]


def item_context_feasible(item, context, definitions, action_checker):
    """Working policy: every member must pass the SAME context check.

    This is a shared-context bundle, not a multi-context day schedule.
    The existing single-action check is delegated unchanged.
    """
    return all(action_checker(a, context, definitions)
               for a in normalize_item(item))


def item_in_group(item, group, mapping):
    """Rule used by v7: an item belongs to a group only if every member does.

    Each actID must be present in the mapping and its group must equal the
    requested group. Mixed-group combinations and items with an unmapped
    actID belong to no group; they are never assigned by first member.
    v7 applies this same rule to candidates and to the grouped reference.
    Whether this rule was approved by the mentor is not recorded here.
    """
    return all(a in mapping and int(mapping[a]) == int(group)
               for a in normalize_item(item))
