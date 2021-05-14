from typing import List, Dict, Tuple, Callable, Any, Optional

import grams.misc as M
from grams.config import DATA_DIR
from grams.kg_data.wikidatamodels import WDProperty, QNode
from grams.main import GRAMS


def update_props(props: List[str]):
    props: Dict[str, QNode] = GRAMS._query_wikidata_entities(props)
    ser = []
    for p in props.values():
        np = WDProperty(p.id, str(p.label), str(p.description), str(p.datatype), [str(s) for s in p.aliases],
                   sorted({stmt.value.as_qnode_id() for stmt in p.props.get("P1647", [])}),
                   sorted({stmt.value.as_qnode_id() for stmt in p.props.get("P1659", [])}),
                   sorted({stmt.value.as_string() for stmt in p.props.get("P1628", [])}),
                   sorted({stmt.value.as_qnode_id() for stmt in p.props.get("P1629", [])}),
                   sorted({stmt.value.as_qnode_id() for stmt in p.props.get("P1696", [])}),
                   sorted({stmt.value.as_qnode_id() for stmt in p.props.get("P31", [])}))
        ser.append(np.serialize())
    M.serialize_byte_lines(ser, DATA_DIR / "new_properties.jl")


if __name__ == '__main__':
    update_props(["P8901"])
