from __future__ import annotations
from operator import itemgetter
from typing import Mapping, Optional, TypeVar, Callable

from kgdata.wikidata.models.wdclass import WDClass
from kgdata.wikidata.models.wdproperty import WDProperty

K = TypeVar("K")


def normalize_probs(
    target_values: list[tuple[K, float]],
    eps: float = 0.001,
    threshold: float = 0.0,
) -> list[tuple[K, float]]:
    """The predicted probabilities can be noisy due to numerical optimizations, i.e., equal variables
    can have slightly different scores. This function groups values that are close
    within the range [-eps, +eps] together, and replace them with the average value
    to reduce the noises.

    Args:
        target_values: probability of grounded values of a predicate
        eps: group values within [-eps, +eps] together
        threshold: remove values with probability less than threshold

    Return:
        A dictionary of normalized probabilities >= threshold
    """
    if eps == 0.0 and threshold == 0.0:
        return target_values

    norm_probs: dict[K, float] = {}
    lst = sorted([x for x in target_values if x[1] >= threshold], key=itemgetter(1))

    if len(lst) == 0:
        return []

    clusters = []
    pivot = 1
    clusters = [[lst[0]]]
    while pivot < len(lst):
        x = lst[pivot - 1][1]
        y = lst[pivot][1]
        if (y - x) <= eps:
            # same clusters
            clusters[-1].append(lst[pivot])
        else:
            # different clusters
            clusters.append([lst[pivot]])
        pivot += 1
    for cluster in clusters:
        avg_prob = sum([x[1] for x in cluster]) / len(cluster)
        for k, _prob in cluster:
            norm_probs[k] = avg_prob
    return sorted(norm_probs.items(), key=itemgetter(1), reverse=True)


def tiebreak(
    obj_and_score: list[tuple[K, float]],
    get_id: Callable[[K], str],
    id2popularity: Mapping[str, int],
    id2ent: Mapping[str, WDClass | WDProperty],
    eps: float = 1e-5,
):
    # decreasing sequence
    assert all(
        obj_and_score[i - 1][1] >= obj_and_score[i][1]
        for i in range(1, len(obj_and_score))
    ), "expect decreasing sequence"
    if len(obj_and_score) == 0:
        return

    best_score = obj_and_score[0][1]
    best_objs = []
    for obj, score in obj_and_score:
        if score < best_score:
            break
        best_objs.append(obj)

    best_idx = 0
    best_ent = id2ent[get_id(best_objs[best_idx])]

    for i in range(1, len(best_objs)):
        ent_id = get_id(best_objs[i])
        if (
            ent_id in best_ent.ancestors
            or id2popularity[ent_id] > id2popularity[best_ent.id]
        ):
            # the counting should be accumulated (the parent should have more count than the children) but
            # it is not, so we manually check it here
            best_idx = i
            best_ent = id2ent[ent_id]

    for i in range(len(best_objs)):
        if i != best_idx:
            # modified in place
            obj_and_score[i] = (obj_and_score[i][0], obj_and_score[i][1] - eps)
