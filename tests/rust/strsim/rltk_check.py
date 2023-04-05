import pandas as pd
import rltk.similarity as sim

testcases = [
    ("MARTHA", "MARHTA"),
    ("DWAYNE", "DUANE"),
    ("DIXON", "DICKSONX"),
]

rows = []
for s1, s2 in testcases:
    rows.append(
        {
            "s1": s1,
            "s2": s2,
            "jaro": sim.jaro_distance(s1, s2),
            "jaro_winkler": sim.jaro_winkler_similarity(s1, s2),
            "hybrid_jaccard_default": sim.hybrid_jaccard_similarity(
                set(s1.split(" ")), set(s2.split(" "))
            ),
        }
    )

df = pd.DataFrame(rows)
print(df)
