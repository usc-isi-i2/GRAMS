from typing import Optional

import orjson

import grams.outputs as O
from grams.inputs.linked_table import LinkedTable
from sm_annotator.annotator import Annotator
from sm_annotator.annotator import Session


class SMVisualizer(Annotator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.app_js_render_fn = "Annotator.renderVizSemModelApp" if self.dev else "Annotator.renderDevVizSemModelApp"

    def visualize(self, id: str, note: str, sm1: O.SemanticModel, sm2: O.SemanticModel,
                  table: Optional[LinkedTable] = None, two_column_layout: bool = False):
        self.session = Session(id, True, note, table, [sm1, sm2])

        cmds = [
            {
                "type": "wait_for_client_ready",
            },
            {
                "type": "set_props",
                "props": {
                    "log": {
                        "note": note,
                    },
                    "graphs": [self.serialize_sm(graph) for graph in self.session.graphs],
                    "currentGraphIndex": 0,
                    "wdOntology": {},
                    "twoColumnLayout": two_column_layout,
                    "entities": {},
                    "assistant": {
                        "id": self.session.table.id
                    },
                }
            }
        ]

        if table is not None:
            cmds[1]['props']['table'] = self.serialize_table_schema()
            cmds.append({
                "type": "exec_func",
                "func": "app.tableFetchData",
                "args": []
            })
        else:
            self.table = None
        self.tunnel.send_msg(orjson.dumps(cmds).decode())
