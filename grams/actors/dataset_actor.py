from dataclasses import dataclass, field
from functools import cached_property
import random
from typing import Literal, Mapping
from grams.actors.db_actor import GramsDBActor
from grams.inputs.linked_table import CandidateEntityId, ExtendedLink, LinkedTable
from kgdata.wikidata.db import WikidataDB
from kgdata.wikidata.models.wdentity import WDEntity
from loguru import logger
from ned.actors.candidate_generation import CanGenActor
from ned.actors.candidate_ranking import CanRankActor
from ned.actors.evaluate_helper import EvalArgs
from osin.integrations.ream import OsinActor
from ream.cache_helper import Cache, CacheArgsHelper
from ream.dataset_helper import DatasetDict, DatasetQuery
from ream.params_helper import NoParams
import serde.json
from slugify import slugify
from sm.dataset import Example, FullTable
from sm.inputs.context import Context
from sm.inputs.link import WIKIDATA, EntityId
from sm.namespaces.wikidata import WikidataNamespace
from sm_datasets.datasets import Datasets
import numpy as np
from tqdm import tqdm


class GramsDatasetActor(OsinActor[str, NoParams]):
    VERSION = 106

    def __init__(self, params: NoParams, db_actor: GramsDBActor):
        super().__init__(params, [db_actor])
        self.db_actor = db_actor
        self.kgns = WikidataNamespace.create()

    @Cache.cls.dir(
        cls=DatasetDict,
        mem_persist=True,
        compression="lz4",
        log_serde_time=True,
    )
    def run(self, dsquery: str) -> DatasetDict[list[Example[LinkedTable]]]:
        parsed_dsquery = DatasetQuery.from_string(dsquery)
        sm_examples = self.load_dataset(parsed_dsquery.dataset)

        infodir = self.get_working_fs().root / (
            f"info_" + slugify(parsed_dsquery.dataset)
        )
        infodir.mkdir(exist_ok=True)

        if parsed_dsquery.shuffle:
            index = list(range(len(sm_examples)))
            random.Random(parsed_dsquery.seed).shuffle(index)

            shuffle_index = [sm_examples[i].table.table.table_id for i in index]
            serde.json.ser(
                shuffle_index,
                infodir / f"shuffle_{parsed_dsquery.seed or 'none'}.json",
            )
            id2e = {e.table.table.table_id: e for e in sm_examples}
            sm_examples = [id2e[i] for i in shuffle_index]

        newexamples: list[Example[LinkedTable]] = []
        for example in sm_examples:
            newexamples.append(
                Example(
                    sms=example.sms,
                    table=LinkedTable.from_full_table(example.table),
                )
            )

        return parsed_dsquery.select(newexamples)

    @Cache.pickle.file(mem_persist=True, compression="lz4")
    def load_dataset(self, dataset: str) -> list[Example[FullTable]]:
        ds = Datasets()
        db = self.db_actor.db
        wdentities = db.get_auto_cached_entities(None)
        examples = getattr(ds, dataset)()
        examples = ds.fix_redirection(
            examples, wdentities, WikidataDB(db.data_dir).wdredirections, self.kgns
        )
        return examples


@dataclass
class GramsELParams:
    topk: int = field(
        default=5,
        metadata={
            "help": "keeping maximum top k candidates for each cell",
        },
    )
    use_oracle: bool = field(
        default=False,
        metadata={
            "help": "use oracle candidates",
        },
    )
    add_gold: Literal["no", "when-not-topk", "always"] = field(
        default="no",
        metadata={
            "help": "add gold entities. options:"
            "- no: never"
            "- when-not-topk: when the gold entity is in candidate entities but not in the top k"
            "- always: add the gold entity when it's missing",
        },
    )
    skip_meta_entities: set[str] = field(
        default_factory=set,
        metadata={
            "help": "skip meta entities such as disambiguation page (Q4167410)",
        },
    )


class GramsELDatasetActor(OsinActor[str, GramsELParams]):
    """
    CHANGELOG:
    - 102: Each cell contains maximum one single link"""

    VERSION = 106

    def __init__(
        self,
        params: GramsELParams,
        db_actor: GramsDBActor,
        dataset_actor: GramsDatasetActor,
        cangen_actor: CanGenActor,
        canrank_actor: CanRankActor,
    ):
        super().__init__(params, [db_actor, dataset_actor, cangen_actor, canrank_actor])
        self.db_actor = db_actor
        self.cangen_actor = cangen_actor
        self.canrank_actor = canrank_actor
        self.dataset_actor = dataset_actor

    def get_provenance(self):
        if self.params.use_oracle:
            return "oracle"
        return self.canrank_actor.get_provenance()

    @Cache.cls.dir(
        cls=DatasetDict,
        cache_self_args=CacheArgsHelper.gen_cache_self_args(get_provenance),
        mem_persist=True,
        compression="lz4",
        log_serde_time=True,
    )
    def run_dataset(self, dsquery: str) -> DatasetDict[list[Example[LinkedTable]]]:
        dsdict = self.dataset_actor.run(dsquery)
        output: DatasetDict[list[Example[LinkedTable]]] = DatasetDict(
            dsdict.name, {}, self.get_provenance()
        )

        if self.params.use_oracle:
            for name, examples in dsdict.items():
                newexamples: list[Example[LinkedTable]] = []
                for ex in examples:
                    newlinks = ex.table.links.shallow_copy()
                    for ri, ci, links in ex.table.links.enumerate_flat_iter():
                        if len(links) == 0:
                            continue
                        link = ExtendedLink(
                            start=0,
                            end=len(ex.table.table[ri, ci]),
                            url=";".join(urls)
                            if (urls := [l.url for l in links if l.url is not None])
                            else None,
                            entities=[eid for link in links for eid in link.entities],
                            candidates=[
                                CandidateEntityId(eid, 1.0)
                                for eid in sorted(
                                    {eid1 for link in links for eid1 in link.entities}
                                )
                            ],
                        )
                        newlinks[ri, ci] = [link]

                    newexamples.append(
                        Example(
                            sms=ex.sms,
                            table=LinkedTable(
                                table=ex.table.table,
                                context=ex.table.context,
                                links=newlinks,
                            ),
                        )
                    )
                output[name] = newexamples
            return output

        cg_dsdict = self.cangen_actor.run_dataset(dsquery)
        cr_dsdict = self.canrank_actor.run_dataset(dsquery)

        for name, examples in dsdict.items():
            candidates = cg_dsdict[name]
            candidates = candidates.replace("score", cr_dsdict[name].score)

            topk_cans = candidates.top_k_candidates(self.params.topk)
            newexamples: list[Example[LinkedTable]] = []
            for example in tqdm(
                examples, desc=f"adding linked entities to examples in {name}"
            ):
                # populate the candidates to links
                # because the entity linking method assumes one link per cell
                # if there is multiple gold links in a cell, we will reduce it to
                # just one link with the ground-truth containing all entities of
                # links in the cell.
                table = example.table
                wdentities = self.db_actor.db.get_auto_cached_entities(table)
                # the candidates inside table.links are always empty because grams dataset actor
                # create a linked table from a full table which does not have candidates
                newlinks = table.links.shallow_copy()
                context = Context(
                    page_title=table.context.page_title,
                    page_url=table.context.page_url,
                    page_entities=[],
                    content_hierarchy=table.context.content_hierarchy,
                )
                for ri, ci, links in table.links.enumerate_flat_iter():
                    cell = table.links[ri, ci]
                    tindex = topk_cans.index[table.id][2]
                    if ci in tindex and ri in tindex[ci][2]:
                        rstart, rend = tindex[ci][2][ri]
                    else:
                        rstart, rend = 0, 0

                    if len(links) == 1:
                        link = ExtendedLink(
                            start=0,
                            end=len(cell),
                            url=links[0].url,
                            entities=links[0].entities.copy(),
                            candidates=[],
                        )
                    elif len(links) > 1:
                        link = ExtendedLink(
                            start=0,
                            end=len(cell),
                            url=";".join(urls)
                            if (urls := [l.url for l in links if l.url is not None])
                            else None,
                            entities=[eid for link in links for eid in link.entities],
                            candidates=[],
                        )
                    else:
                        if rend > rstart:
                            # no gold links but we have some candidates
                            # so we need to create a new link
                            link = ExtendedLink(
                                start=0,
                                end=len(cell),
                                url=None,
                                entities=[],
                                candidates=[],
                            )
                        else:
                            # no gold links and no candidates
                            continue

                    can_ids = topk_cans.id[rstart:rend]
                    can_scores = topk_cans.score[rstart:rend]
                    link.candidates = [
                        CandidateEntityId(EntityId(can_id, WIKIDATA), can_score)
                        for can_id, can_score in zip(can_ids, can_scores)
                    ]
                    if self.params.add_gold == "when-not-topk":
                        gold_ents = set(link.entities)
                        all_can_ids = candidates.get_cell_candidates(table.id, ri, ci)
                        # if not np.isin(can_ids, gold_ents).any() and :
                        if not any(id in gold_ents for id in can_ids):
                            matches = [
                                i
                                for i, id in enumerate(all_can_ids.id)
                                if id in gold_ents
                            ]
                            if len(matches) > 0:
                                link.candidates.extend(
                                    [
                                        CandidateEntityId(
                                            EntityId(all_can_ids.id[i], WIKIDATA),
                                            all_can_ids.score[i],
                                        )
                                        for i in matches
                                    ]
                                )
                    elif self.params.add_gold == "always":
                        gold_ents = set(link.entities)
                        if all(
                            can.entity_id not in gold_ents for can in link.candidates
                        ):
                            if len(link.candidates) == 0:
                                score = 0.5
                            else:
                                score = min(c.probability for c in link.candidates)
                            link.candidates.extend(
                                [
                                    CandidateEntityId(EntityId(eid, WIKIDATA), score)
                                    for eid in gold_ents
                                ]
                            )
                    else:
                        assert self.params.add_gold == "no"

                    link.candidates = [
                        can
                        for can in link.candidates
                        if not self.is_metadata_entity(can.entity_id, wdentities)
                    ]
                    newlinks[ri, ci] = [link]

                newexamples.append(
                    Example(
                        sms=example.sms,
                        table=LinkedTable(
                            table=table.table,
                            context=context,
                            links=newlinks,
                        ),
                    )
                )

            output[name] = newexamples
        return output

    def evaluate(self, evalargs: EvalArgs):
        for dsquery in evalargs.dsqueries:
            dsquery_p = DatasetQuery.from_string(dsquery)
            dsdict = self.dataset_actor.run_dataset(dsquery)
            ann_dsdict = self.run_dataset(dsquery)

            for name, examples in dsdict.items():
                dsname = dsquery_p.get_query(name)
                with self.new_exp_run(
                    dataset=dsname, exprun_type=evalargs.exprun_type
                ) as exprun:
                    pass

    def is_metadata_entity(self, entity_id: str, wdentities: Mapping[str, WDEntity]):
        """Test if an entity is a metadata entity or instance of a metadata entity."""
        return entity_id not in self.params.skip_meta_entities and (
            entity_id in wdentities
            and any(
                stmt.value.as_entity_id_safe() in self.params.skip_meta_entities
                for stmt in wdentities[entity_id].props.get("P31", [])
            )
        )
