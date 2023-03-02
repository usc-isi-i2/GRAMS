from __future__ import annotations
from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Optional
from grams.actors.grams_infdata_actor import InfData, InfDataDatasetDict
from grams.algorithm.inferences_v2.features.inf_feature import InfFeature
from grams.evaluation.autolabeler import AutoLabeler
from grams.evaluation.evaluator import Evaluator
from grams.inputs.linked_table import LinkedTable
from ned.candidate_ranking.helpers.dataset import MyDataset, MyDatasetDict
from ned.candidate_ranking.helpers.training_helper import ModifiedTrainingArguments

from ned.models_and_training.ensemble_classifier import (
    EnsembleClassifier,
    EnsembleClassifierArgs,
)
from ned.models_and_training.sklearn_classifier import sklearn_train
import numpy as np
from ream.dataset_helper import DatasetDict
from sm.dataset import Example


@dataclass
class IndConfig:
    ensemble_args: EnsembleClassifierArgs = field(
        default_factory=EnsembleClassifierArgs,
    )
    train_args: ModifiedTrainingArguments = field(
        default_factory=ModifiedTrainingArguments,
    )


DSQuery = Callable[
    [str], tuple[DatasetDict[list[InfFeature]], DatasetDict[list[Example[LinkedTable]]]]
]


class IndModel:
    VERSION = 100

    def __init__(self, args: IndConfig):
        self.args = args
        self.rel_model = EnsembleClassifier(args.ensemble_args)
        self.type_model = EnsembleClassifier(args.ensemble_args)

    def train(
        self,
        trainargs: ModifiedTrainingArguments,
        query_dataset: DSQuery,
        evaluator: Evaluator,
    ):
        autolabeler = AutoLabeler(evaluator)
        self.type_model = sklearn_train(
            self.type_model,
            GenTypeData(query_dataset, autolabeler),
            trainargs,
            metrics=["precision", "recall", "f1"],
        )

    def predict(self, feat: InfFeature):
        ds = GenTypeData.get_dataset([feat], None, None)
        edge_probs = feat.edge_features.freq_over_row
        node_probs = self.type_model.predict_proba(
            **{name: ds.examples[name] for name in self.type_model.EVAL_ARGS}
        )

        return edge_probs, node_probs


class GenTypeData:
    def __init__(self, query_dataset: DSQuery, autolabeler: AutoLabeler):
        self.query_dataset = query_dataset
        self.autolabeler = autolabeler

    @classmethod
    def get_dataset(
        cls,
        feat_ds: list[InfFeature],
        ex_ds: Optional[list[Example[LinkedTable]]],
        autolabeler: Optional[AutoLabeler],
    ) -> MyDataset:
        ids = []
        types = []
        labels = []
        features = [[], [], [], [], [], []]

        for i, inf_feat in enumerate(feat_ds):
            ufeat = inf_feat.node_features
            id = ufeat.node

            ids.append(id)
            types.append(ufeat.type)

            if autolabeler is not None:
                example = ex_ds[i]  # type: ignore
                type = [(id[i], inf_feat.idmap.im(k)) for i, k in enumerate(ufeat.type)]
                labels.append(autolabeler.label_example_types(type, example))

            features[0].append(ufeat.freq_over_row)
            features[1].append(ufeat.freq_over_ent_row)
            features[2].append(ufeat.extended_freq_over_row)
            features[3].append(ufeat.extended_freq_over_ent_row)
            features[4].append(ufeat.freq_discovered_prop_over_row)
            features[5].append(ufeat.type_distance)

        allfeat = np.zeros(
            (sum(len(x) for x in features[0]), len(features)), dtype=np.float64
        )
        for i, feat in enumerate(features):
            start = 0
            for x in feat:
                end = start + len(x)
                allfeat[start:end, i] = x
                start = end

        examples = {
            "id": np.concatenate(ids),
            "type": np.concatenate(types),
            "features": allfeat,
        }
        if autolabeler is not None:
            examples["label"] = np.concatenate(labels)
        return MyDataset(examples=examples)

    def __call__(
        self,
        method: EnsembleClassifier,
        dataset_query: str,
        for_training: bool,
    ) -> DatasetDict[MyDataset]:
        feat_dsdict, example_dsdict = self.query_dataset(dataset_query)

        out: DatasetDict[MyDataset] = DatasetDict(name=feat_dsdict.name, subsets={})
        for name, feat_ds in feat_dsdict.items():
            ex_ds = example_dsdict[name]
            out[name] = self.get_dataset(feat_ds, ex_ds, self.autolabeler)
        return out


@dataclass
class IndModelData:
    rel_features: np.ndarray
    rel_label: np.ndarray
    type_features: np.ndarray
    type_label: np.ndarray

    @staticmethod
    def from_inf_features(feat: InfFeature) -> IndModelData:
        assert False
