import glob
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Tuple, Callable, Any, Optional
import grams.misc as M
from grams.config import ROOT_DIR


def clean_table_linker(infile, outfile):
    rows = M.deserialize_csv(infile)
    cname2idx = {v: k for k, v in enumerate(rows[0])}

    cells = defaultdict(dict)
    for row in rows[1:]:
        ci = int(row[cname2idx['column']])
        ri = int(row[cname2idx['row']])
        gt_ent = row[cname2idx['GT_kg_id']]
        pred_ent = row[cname2idx['kg_id']]
        score = row[cname2idx['siamese_pred']]

        if (ri, ci) in cells:
            assert cells[ri, ci]['gt'] == gt_ent
        cells[ri, ci]['gt'] = gt_ent
        if 'pred_ent' not in cells[ri, ci]:
            cells[ri, ci]['pred_ent'] = []
        cells[ri, ci]['pred_ent'].append(f"{pred_ent}:{score}")

    links = []
    for (ri, ci), o in cells.items():
        links.append([ri, ci, o['gt']] + o['pred_ent'])
    M.serialize_csv(links, outfile, delimiter="\t")


if __name__ == '__main__':
    for infile in glob.glob(str(ROOT_DIR / "examples/t2dv2/tables/*.table_linker.csv")):
        infile = Path(infile)
        outfile = infile.parent / f"{infile.name.split('.')[0]}.candidates.tsv"
        clean_table_linker(infile, outfile)