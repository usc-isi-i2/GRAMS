import glob
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import sm.misc as M
from grams.config import ROOT_DIR


def clean_table_linker(
    infile, outfile, ext_gt: Optional[Dict[Tuple[int, int], str]] = None
):
    rows = M.deserialize_csv(infile)
    cname2idx = {v: k for k, v in enumerate(rows[0])}

    cells = defaultdict(dict)
    for row in rows[1:]:
        ci = int(row[cname2idx["column"]])
        ri = int(row[cname2idx["row"]])
        if "GT_kg_id" in cname2idx:
            gt_ent = row[cname2idx["GT_kg_id"]]
        elif ext_gt is not None and (ri, ci) in ext_gt:
            gt_ent = ext_gt[(ri, ci)]
        else:
            gt_ent = None

        pred_ent = row[cname2idx["kg_id"]]
        score = float(row[cname2idx["siamese_prediction"]])

        if (ri, ci) in cells:
            assert cells[ri, ci]["gt"] == gt_ent
        cells[ri, ci]["gt"] = gt_ent

        if "pred_ent" not in cells[ri, ci]:
            cells[ri, ci]["pred_ent"] = []
        cells[ri, ci]["pred_ent"].append(f"{pred_ent}:{score:.9f}")

    links = []
    for (ri, ci), o in cells.items():
        links.append([ri, ci, o["gt"]] + o["pred_ent"])
    M.serialize_csv(links, outfile, delimiter="\t")


if __name__ == "__main__":
    for infile in glob.glob(str(ROOT_DIR / "examples/t2dv2/tables/*.table_linker.csv")):
        infile = Path(infile)
        outfile = infile.parent / f"{infile.name.split('.')[0]}.candidates.tsv"
        clean_table_linker(infile, outfile)
