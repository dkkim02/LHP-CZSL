"""Build data/logic_rules_mit.json for C′ axis (LOGICZSL-style logic rule grounding).

Two rule families:
1. attr_mutex_groups: groups of attrs that are mutually exclusive states of the same property.
   Hand-curated from MIT-States 115 attrs; conservative — only groups with clear mutex semantics
   so the model is not penalized for plausible co-occurrence.
2. obj_incompat_attrs: per-obj list of attrs LLM judged implausible. Reuses Gemini-scored
   data/feasibility_mit.json with threshold score <= INCOMPAT_THRESHOLD.

Output (UTF-8 JSON):
  {
    "meta": {...},
    "attr_index": {attr: i, ...},        # for fast index lookup in training
    "obj_index": {obj: j, ...},
    "attr_mutex_groups": {group_name: [attr, ...], ...},
    "attr_mutex_membership": [group_id_or_-1 per attr],   # array length N_attr
    "obj_incompat_attrs": {obj: [attr, ...], ...},        # categorical from feasibility threshold
  }

Reason for reusing feasibility json: scalar score sealed mechanism was logit-additive bias
(`logit_c += w·s(a,o)`). C′ uses the SAME data but as a categorical incompat indicator on
attr_logits posterior — different operating surface.
"""
import json
import sys
from pathlib import Path

# Repo paths
REPO = Path(__file__).resolve().parents[1]
FEASIBILITY_JSON = REPO / "data" / "feasibility_mit.json"
OUT_JSON = REPO / "data" / "logic_rules_mit.json"

# Threshold: feasibility score in [1, 10]; <= 3 means "implausible"
INCOMPAT_THRESHOLD = 3

# MIT-States 115 attrs (matches build_descriptions_claude.py)
ATTRS_LIST = [
    "ancient","barren","bent","blunt","bright","broken","browned","brushed","burnt","caramelized",
    "chipped","clean","clear","closed","cloudy","cluttered","coiled","cooked","cored","cracked",
    "creased","crinkled","crumpled","crushed","curved","cut","damp","dark","deflated","dented",
    "diced","dirty","draped","dry","dull","empty","engraved","eroded","fallen","filled","foggy",
    "folded","frayed","fresh","frozen","full","grimy","heavy","huge","inflated","large","lightweight",
    "loose","mashed","melted","modern","moldy","molten","mossy","muddy","murky","narrow","new","old",
    "open","painted","peeled","pierced","pressed","pureed","raw","ripe","ripped","rough","ruffled",
    "runny","rusty","scratched","sharp","shattered","shiny","short","sliced","small","smooth",
    "spilled","splintered","squished","standing","steaming","straight","sunny","tall","thawed",
    "thick","thin","tight","tiny","toppled","torn","unpainted","unripe","upright","verdant","viscous",
    "weathered","wet","whipped","wide","wilted","windblown","winding","worn","wrinkled","young",
]

# Hand-curated mutex groups. Only include groups where mutex is clearly satisfied within MIT-States
# semantics (an object cannot be both states at the same time on the same image instance).
# Some attrs (e.g., bent, broken) are intentionally NOT grouped because damage states can co-occur.
MUTEX_GROUPS = {
    "moisture":            ["wet", "damp", "dry"],
    "ripeness":            ["ripe", "unripe"],
    "cook_doneness":       ["raw", "cooked", "burnt"],
    "freshness":           ["fresh", "wilted", "moldy"],
    "thermal_state":       ["frozen", "thawed", "melted", "molten"],
    "fullness":            ["empty", "full", "filled"],
    "inflation":           ["deflated", "inflated"],
    "cleanliness":         ["clean", "dirty", "grimy", "muddy"],
    "era":                 ["ancient", "old", "modern", "new"],
    "age":                 ["young", "old"],
    "size":                ["tiny", "small", "large", "huge"],
    "height":              ["short", "tall"],
    "width":               ["narrow", "wide"],
    "thickness":           ["thin", "thick"],
    "weight_property":     ["heavy", "lightweight"],
    "sharpness":           ["sharp", "blunt"],
    "tightness":           ["tight", "loose"],
    "surface_smoothness":  ["rough", "smooth"],
    "clarity":             ["clear", "cloudy", "foggy", "murky"],
    "openness":            ["open", "closed"],
    "orientation":         ["standing", "fallen", "toppled", "upright"],
    "paint_state":         ["painted", "unpainted"],
    "light_level":         ["bright", "dark"],
    "gloss":               ["shiny", "dull"],
    "fluidity":            ["runny", "viscous"],
}


def build():
    attr_index = {a: i for i, a in enumerate(ATTRS_LIST)}

    # Validate mutex group attrs are all in ATTRS_LIST
    for gname, attrs in MUTEX_GROUPS.items():
        for a in attrs:
            if a not in attr_index:
                raise ValueError(f"mutex group '{gname}' has unknown attr '{a}'")

    # Build per-attr group membership (group_id or -1)
    group_names = list(MUTEX_GROUPS.keys())
    attr_mutex_membership = [-1] * len(ATTRS_LIST)
    for gid, gname in enumerate(group_names):
        for a in MUTEX_GROUPS[gname]:
            ai = attr_index[a]
            # An attr can belong to multiple semantic groups (e.g., "old" ∈ era AND age).
            # Conservative: assign first group; rest are ignored by the simple loss.
            # The loss formulation treats one group per attr (the first hit).
            if attr_mutex_membership[ai] == -1:
                attr_mutex_membership[ai] = gid

    # Load feasibility json and threshold
    with open(FEASIBILITY_JSON, "r", encoding="utf-8") as f:
        feas = json.load(f)
    print(f"feasibility provider: {feas['meta'].get('provider')}", file=sys.stderr)
    print(f"feasibility entries: {len(feas['scores'])}", file=sys.stderr)

    obj_set = set()
    obj_incompat = {}
    score_dist = {}
    n_null_score = 0
    for entry in feas["scores"]:
        a, o, raw_s = entry["attr"], entry["obj"], entry.get("score")
        obj_set.add(o)
        if raw_s is None:
            n_null_score += 1
            continue
        s = int(raw_s)
        score_dist[s] = score_dist.get(s, 0) + 1
        if s <= INCOMPAT_THRESHOLD:
            obj_incompat.setdefault(o, []).append(a)
    if n_null_score:
        print(f"  skipped {n_null_score} entries with null score", file=sys.stderr)

    objs_list = sorted(obj_set)
    obj_index = {o: i for i, o in enumerate(objs_list)}

    # Coverage stats
    n_attr_grouped = sum(1 for x in attr_mutex_membership if x != -1)
    n_obj_with_incompat = len(obj_incompat)
    total_incompat = sum(len(v) for v in obj_incompat.values())

    out = {
        "meta": {
            "source_feasibility": str(FEASIBILITY_JSON.name),
            "incompat_threshold": INCOMPAT_THRESHOLD,
            "feasibility_provider": feas["meta"].get("provider"),
            "n_attrs": len(ATTRS_LIST),
            "n_objs": len(objs_list),
            "n_mutex_groups": len(group_names),
            "n_attrs_grouped": n_attr_grouped,
            "n_objs_with_incompat": n_obj_with_incompat,
            "total_incompat_entries": total_incompat,
            "feasibility_score_distribution": score_dist,
        },
        "attr_index": attr_index,
        "obj_index": obj_index,
        "mutex_group_names": group_names,
        "attr_mutex_groups": MUTEX_GROUPS,
        "attr_mutex_membership": attr_mutex_membership,
        "obj_incompat_attrs": obj_incompat,
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"wrote {OUT_JSON}", file=sys.stderr)
    print(f"  attrs grouped: {n_attr_grouped}/{len(ATTRS_LIST)}", file=sys.stderr)
    print(f"  objs with incompat: {n_obj_with_incompat}/{len(objs_list)}", file=sys.stderr)
    print(f"  total incompat entries: {total_incompat}", file=sys.stderr)
    print(f"  score distribution: {sorted(score_dist.items())}", file=sys.stderr)


if __name__ == "__main__":
    build()
