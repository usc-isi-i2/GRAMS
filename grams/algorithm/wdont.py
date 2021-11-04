from enum import IntEnum
from typing import Dict

from kgdata.wikidata.models import QNode, WDClass, WDProperty
from sm.misc import identity_func


class OutOfNamespace(Exception):
    pass


class WDOnt:
    STATEMENT_URI = "http://wikiba.se/ontology#Statement"
    STATEMENT_REL_URI = "wikibase:Statement"

    def __init__(
        self,
        qnodes: Dict[str, QNode],
        wdclasses: Dict[str, WDClass],
        wdprops: Dict[str, WDProperty],
    ):
        self.qnodes = qnodes
        self.wdclasses = wdclasses
        self.wdprops = wdprops
        self.get_qid_fn = {
            "Q": identity_func,
            "q": identity_func,
            # http
            "h": self.get_qnode_id,
        }
        self.get_pid_fn = {
            "P": identity_func,
            "p": identity_func,
            "h": self.get_prop_id,
        }

    @classmethod
    def is_uri_statement(cls, uri: str):
        return uri == "http://wikiba.se/ontology#Statement"

    @classmethod
    def is_uri_dummy_class(cls, uri: str):
        return uri == "http://wikiba.se/ontology#DummyClassForInversion"

    @classmethod
    def is_uri_property(cls, uri: str):
        return uri.startswith(f"http://www.wikidata.org/prop/")

    @classmethod
    def is_uri_qnode(cls, uri: str):
        return uri.startswith("http://www.wikidata.org/entity/")

    @classmethod
    def get_qnode_id(cls, uri: str):
        if not cls.is_uri_qnode(uri):
            raise OutOfNamespace(f"{uri} is not in wikidata qnode namespace")
        return uri.replace("http://www.wikidata.org/entity/", "")

    @classmethod
    def get_qnode_uri(cls, qnode_id: str):
        return f"http://www.wikidata.org/entity/{qnode_id}"

    @classmethod
    def get_prop_id(cls, uri: str):
        if not cls.is_uri_property(uri):
            raise OutOfNamespace(f"{uri} is not in wikidata property namespace")
        return uri.replace(f"http://www.wikidata.org/prop/", "")

    @classmethod
    def get_prop_uri(cls, pid: str):
        return f"http://www.wikidata.org/prop/{pid}"

    def get_qnode_label(self, uri_or_id: str):
        qid = self.get_qid_fn[uri_or_id[0]](uri_or_id)
        if qid in self.wdclasses:
            return f"{self.wdclasses[qid].label} ({qid})"
        return f"{self.qnodes[qid].label} ({qid})"

    def get_pnode_label(self, uri_or_id: str):
        pid = self.get_pid_fn[uri_or_id[0]](uri_or_id)
        # TODO: fix me! should not do this
        if pid not in self.wdprops:
            return pid
        return f"{self.wdprops[pid].label} ({pid})"
