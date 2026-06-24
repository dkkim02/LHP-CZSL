"""Build data/hardneg_pairs_mit.json for D′ axis (LLM-guided semantic hard-negative mining).

For each of the 1262 MIT-States train compositions, produce up to K=5 visually-confusable
hard-negative compositions. "Visually confusable" is operationalized via two Claude-curated
semantic similarity sources (the LLM-in-conversation lever of D′, §20-5):

  ATTR_SIM_GROUPS : groups of attributes that produce similar appearance (same perceptual
                    property / state family, or shared damage/texture look). Reuses the C′
                    mutex groups (§20-4-1) plus non-mutex visual-confusion groups (damage,
                    creasing, corrosion, ...). An attr's siblings are all co-group members.
  OBJ_FAMILIES    : object families that look alike (shape / material / scene). Many-to-many:
                    an object's siblings are the union of all families it belongs to.

Per anchor (a, o), candidate hard-negs are drawn from three tiers, keeping only comps that
are valid TRAIN pairs and excluding the anchor itself:
  tier1 (same obj, sibling attr)  : (a', o)  for a' in attr_sim(a)\\{a}     — "same object, look-alike state"
  tier2 (same attr, sibling obj)  : (a, o')  for o' in obj_fam(o)\\{o}     — "same state, look-alike object"
  tier3 (cross-primitive)         : (a', o') a'≠a, o'≠o, both siblings    — D′'s distinguishing lever

Selection (deterministic): quota 2×tier1 + 2×tier2 + 1×tier3, then backfill in tier order to
reach K. tier3 is reserved a slot so cross-primitive confusions (the mechanism that separates
D′ from the sealed syntactic hardpair, §17-9) are represented when available.

Output (UTF-8 JSON):
  {
    "meta": {...},
    "hard_neg": { "attr obj": ["a1 o1", ..., "aK oK"], ... }   # only train comps, <= K each
  }
"""
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TRAIN_PAIRS = Path("/home/student/dongki/DPAS/data/mit-states/compositional-split-natural/train_pairs.txt")
OUT_JSON = REPO / "data" / "hardneg_pairs_mit.json"

K = 5

# ---------------------------------------------------------------------------
# Attribute visual-similarity groups (Claude-curated). Superset of C′ mutex groups
# (build_logic_rules.py MUTEX_GROUPS) plus non-mutex visual-confusion families. Overlap
# across groups is intentional — an attr's siblings = union of all its groups.
# ---------------------------------------------------------------------------
ATTR_SIM_GROUPS = {
    # --- C′ mutex groups (same property, different state) ---
    "moisture":           ["wet", "damp", "dry"],
    "ripeness":           ["ripe", "unripe"],
    "cook_doneness":      ["raw", "cooked", "burnt", "browned", "caramelized"],
    "freshness":          ["fresh", "wilted", "moldy"],
    "thermal_state":      ["frozen", "thawed", "melted", "molten", "steaming"],
    "fullness":           ["empty", "full", "filled"],
    "inflation":          ["deflated", "inflated"],
    "cleanliness":        ["clean", "dirty", "grimy", "muddy"],
    "era":                ["ancient", "old", "modern", "new"],
    "age":                ["young", "old"],
    "size":               ["tiny", "small", "large", "huge"],
    "height":             ["short", "tall"],
    "width":              ["narrow", "wide"],
    "thickness":          ["thin", "thick"],
    "weight_property":    ["heavy", "lightweight"],
    "sharpness":          ["sharp", "blunt"],
    "tightness":          ["tight", "loose"],
    "surface_smoothness": ["rough", "smooth"],
    "clarity":            ["clear", "cloudy", "foggy", "murky"],
    "openness":           ["open", "closed"],
    "orientation":        ["standing", "fallen", "toppled", "upright"],
    "paint_state":        ["painted", "unpainted", "engraved"],
    "light_level":        ["bright", "dark", "sunny"],
    "gloss":              ["shiny", "dull"],
    "fluidity":           ["runny", "viscous"],
    # --- non-mutex visual-confusion groups (similar appearance, may co-occur) ---
    "damage_break":       ["broken", "cracked", "chipped", "dented", "shattered", "splintered",
                           "scratched", "bent"],
    "tear":               ["torn", "ripped", "frayed", "cut"],
    "crush":              ["crushed", "squished", "crumpled", "mashed", "pressed"],
    "crease":             ["creased", "crinkled", "crumpled", "folded", "wrinkled", "ruffled", "draped"],
    "slice_prep":         ["cut", "sliced", "diced", "peeled", "cored", "mashed", "pureed", "whipped"],
    "corrosion":          ["rusty", "eroded", "weathered", "worn", "mossy", "moldy", "browned"],
    "curve":              ["coiled", "curved", "winding", "bent", "straight"],
    "spill":              ["spilled", "runny", "wet", "splintered"],
    "barren_state":       ["barren", "eroded", "verdant"],
    "pierced_state":      ["pierced", "cracked", "cut"],
    "windblown_state":    ["windblown", "ruffled", "draped"],
    "surface_finish":     ["brushed", "smooth", "shiny", "dull", "rough"],
    "occupancy":          ["cluttered", "full", "filled", "empty"],
}

# ---------------------------------------------------------------------------
# Object visual/taxonomic families (Claude-curated). Many-to-many; an obj's siblings =
# union of all families containing it.
# ---------------------------------------------------------------------------
OBJ_FAMILIES = {
    "fruit":          ["apple", "banana", "berry", "fig", "fruit", "lemon", "orange", "pear",
                       "persimmon", "tomato"],
    "produce_veg":    ["bean", "garlic", "nut", "potato", "vegetable", "salad", "tomato"],
    "bakery_sweet":   ["bread", "cake", "candy", "chocolate", "cookie", "pie", "pizza",
                       "sandwich", "sugar", "butter"],
    "cooked_food":    ["beef", "chicken", "cheese", "eggs", "fish", "meat", "pasta", "salmon",
                       "sauce", "seafood", "soup", "salad", "paste"],
    "drink":          ["coffee", "milk", "tea", "water", "oil", "soup"],
    "animal":         ["animal", "bear", "cat", "dog", "elephant", "horse", "iguana", "snake",
                       "tiger", "fish", "chicken"],
    "building":       ["building", "castle", "church", "house", "library", "tower", "town", "city"],
    "architecture":   ["bridge", "column", "gate", "fence", "wall", "roof", "ceiling", "floor",
                       "deck", "steps", "door", "window", "handle", "frame", "well"],
    "room_space":     ["basement", "bathroom", "kitchen", "room", "garage", "shower"],
    "vehicle":        ["bike", "boat", "bus", "car", "truck"],
    "water_body":     ["bay", "beach", "coast", "creek", "lake", "ocean", "pond", "pool", "river",
                       "sea", "shore", "stream", "wave", "water"],
    "terrain":        ["boulder", "canyon", "cave", "cliff", "desert", "field", "ground", "island",
                       "mountain", "rock", "sand", "stone", "valley", "dirt", "mud"],
    "place_path":     ["road", "highway", "street", "trail", "farm", "garden", "forest", "jungle",
                       "bridge"],
    "plant":          ["branch", "bush", "flower", "leaf", "log", "moss", "palm", "plant",
                       "redwood", "roots", "tree", "tulip", "nest"],
    "sky_gas":        ["cloud", "smoke", "sky", "lightning", "dust", "foam", "bubble"],
    "fire":           ["fire", "flame", "candle", "lightning"],
    "metal":          ["aluminum", "brass", "bronze", "copper", "lead", "metal", "steel", "coal"],
    "coin":           ["coin", "penny"],
    "stone_mineral":  ["boulder", "rock", "stone", "granite", "concrete", "clay", "ceramic",
                       "tile", "glass"],
    "gem":            ["diamond", "gemstone"],
    "textile":        ["cotton", "fabric", "silk", "velvet", "wool", "carpet", "mat"],
    "cordage":        ["cable", "chains", "cord", "hose", "ribbon", "rope", "thread", "wire", "tube"],
    "clothing":       ["armor", "belt", "coat", "dress", "hat", "jacket", "pants", "shirt",
                       "shoes", "shorts", "tie", "clothes", "glasses"],
    "frozen_white":   ["ice", "snow", "glass", "foam", "cloud", "sugar"],
    "coating":        ["paint", "wax", "oil", "paste", "sauce"],
    "jewelry":        ["bracelet", "jewelry", "necklace", "ring", "diamond", "gemstone"],
    "container":      ["bag", "basket", "bottle", "bowl", "box", "bucket", "pot", "plate",
                       "envelope", "tube"],
    "furniture":      ["bed", "cabinet", "chair", "desk", "table", "frame", "furniture"],
    "blade":          ["blade", "knife", "sword"],
    "machine_part":   ["gear", "screw", "wheel", "tire", "fan", "drum", "vacuum", "key"],
    "electronics":    ["camera", "clock", "computer", "keyboard", "laptop", "phone", "lightbulb"],
    "paper":          ["book", "card", "envelope", "newspaper", "paper"],
    "round_obj":      ["ball", "balloon", "bubble", "ring", "coin"],
    "material_misc":  ["plastic", "rubber", "wax", "wood", "glass", "foam"],
    "surface":        ["mirror", "glass", "window", "tile", "mat", "carpet"],
    "natural_obj":    ["shell", "nest", "branch", "roots", "log"],
    "toy_misc":       ["toy", "drum", "ball", "balloon"],
}


def build_sibling_map(groups):
    sib = {}
    for members in groups.values():
        for m in members:
            sib.setdefault(m, set()).update(members)
    for m in sib:
        sib[m].discard(m)
    return sib


def build():
    # Train comps
    train_pairs = []
    with open(TRAIN_PAIRS, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            a, o = line.split()
            train_pairs.append((a, o))
    train_set = set(train_pairs)
    attrs = sorted({a for a, _ in train_pairs})
    objs = sorted({o for _, o in train_pairs})

    attr_sib = build_sibling_map(ATTR_SIM_GROUPS)
    obj_sib = build_sibling_map(OBJ_FAMILIES)

    # Validate curated vocab against dataset vocab
    unknown_attrs = sorted({a for a in attr_sib if a not in attrs})
    unknown_objs = sorted({o for o in obj_sib if o not in objs})
    if unknown_attrs:
        print(f"WARNING: curated attrs not in train vocab: {unknown_attrs}", file=sys.stderr)
    if unknown_objs:
        print(f"WARNING: curated objs not in train vocab: {unknown_objs}", file=sys.stderr)

    def comp_str(a, o):
        return f"{a} {o}"

    hard_neg = {}
    k_dist = Counter()
    tier_counts = Counter()
    objs_no_family = set()
    attrs_no_group = set()

    for (a, o) in train_pairs:
        if a not in attr_sib:
            attrs_no_group.add(a)
        if o not in obj_sib:
            objs_no_family.add(o)

        tier1 = sorted({comp_str(a2, o) for a2 in attr_sib.get(a, ()) if (a2, o) in train_set})
        tier2 = sorted({comp_str(a, o2) for o2 in obj_sib.get(o, ()) if (a, o2) in train_set})
        tier3 = sorted({
            comp_str(a2, o2)
            for a2 in attr_sib.get(a, ())
            for o2 in obj_sib.get(o, ())
            if a2 != a and o2 != o and (a2, o2) in train_set
        })
        anchor = comp_str(a, o)
        for t in (tier1, tier2, tier3):
            if anchor in t:
                t.remove(anchor)

        selected = []
        # quota pass: 2 tier1, 2 tier2, 1 tier3
        for lst, q, name in ((tier1, 2, "t1"), (tier2, 2, "t2"), (tier3, 1, "t3")):
            taken = 0
            for c in lst:
                if len(selected) >= K or taken >= q:
                    break
                if c not in selected:
                    selected.append(c)
                    tier_counts[name] += 1
                    taken += 1
        # backfill in tier order
        for lst, name in ((tier1, "t1"), (tier2, "t2"), (tier3, "t3")):
            for c in lst:
                if len(selected) >= K:
                    break
                if c not in selected:
                    selected.append(c)
                    tier_counts[name] += 1

        if selected:
            hard_neg[anchor] = selected
        k_dist[len(selected)] += 1

    n_with_negs = len(hard_neg)
    total_edges = sum(len(v) for v in hard_neg.values())
    out = {
        "meta": {
            "source_train_pairs": str(TRAIN_PAIRS),
            "n_train_comps": len(train_pairs),
            "n_attrs": len(attrs),
            "n_objs": len(objs),
            "K": K,
            "n_attr_sim_groups": len(ATTR_SIM_GROUPS),
            "n_obj_families": len(OBJ_FAMILIES),
            "n_comps_with_negs": n_with_negs,
            "total_hardneg_edges": total_edges,
            "avg_negs_per_comp": round(total_edges / max(1, n_with_negs), 3),
            "k_distribution": dict(sorted(k_dist.items())),
            "tier_edge_counts": dict(tier_counts),
            "n_objs_no_family": len(objs_no_family),
            "objs_no_family": sorted(objs_no_family),
            "n_attrs_no_group": len(attrs_no_group),
            "attrs_no_group": sorted(attrs_no_group),
        },
        "hard_neg": hard_neg,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"wrote {OUT_JSON}", file=sys.stderr)
    for k, v in out["meta"].items():
        if k in ("objs_no_family", "attrs_no_group"):
            print(f"  {k}: {v}", file=sys.stderr)
        else:
            print(f"  {k}: {v}", file=sys.stderr)


if __name__ == "__main__":
    build()
