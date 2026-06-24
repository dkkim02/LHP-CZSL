"""E axis — LLM taxonomy / structural prior builder.

Curated (Claude in-conversation, §20-6) semantic taxonomy for mit-states:
  - 115 attributes  -> attribute STATE-TYPE groups (semantic kind of state, NOT mutual
    exclusion; distinct from C' mutex groups in logic_rules_mit.json).
  - 245 objects     -> super-categories (fruit, animal, metal, building, ...).

Output: data/taxonomy_mit.json with
  attr_index / obj_index : canonical dataset ordering (for model-side sanity check)
  attr_group_names       : list of group names (index = group id)
  obj_supcat_names       : list of supercat names (index = supcat id)
  attr_group_id          : list[int] len 115, group id per attr (dataset order)
  obj_supcat_id          : list[int] len 245, supcat id per obj (dataset order)

Each attr/obj is assigned to EXACTLY ONE group/supcat; the script asserts full,
non-overlapping coverage against the canonical lists from logic_rules_mit.json.
"""
import json
from pathlib import Path

REPO = Path("/home/student/dongki/LHP-CZSL")
LOGIC = REPO / "data" / "logic_rules_mit.json"
OUT = REPO / "data" / "taxonomy_mit.json"

# ---- canonical ordering source (same attr/obj lists the dataloader produces) ----
_rules = json.load(open(LOGIC, "r", encoding="utf-8"))
ATTR_INDEX = _rules["attr_index"]
OBJ_INDEX = _rules["obj_index"]
ATTRS = [a for a, _ in sorted(ATTR_INDEX.items(), key=lambda x: x[1])]
OBJS = [o for o, _ in sorted(OBJ_INDEX.items(), key=lambda x: x[1])]

# ---------------------------------------------------------------------------
# Attribute STATE-TYPE groups (semantic kind of the state word).
# ---------------------------------------------------------------------------
ATTR_GROUPS = {
    "age_wear":     ["ancient", "old", "new", "modern", "young", "weathered", "worn", "eroded"],
    "size":         ["huge", "large", "small", "tiny", "short", "tall", "wide", "narrow",
                     "thick", "thin", "heavy", "lightweight"],
    "shape":        ["bent", "coiled", "curved", "creased", "crinkled", "crumpled", "draped",
                     "folded", "ruffled", "straight", "winding", "wrinkled"],
    "damage":       ["broken", "cracked", "chipped", "shattered", "dented", "splintered",
                     "frayed", "ripped", "torn", "scratched", "pierced", "crushed", "squished"],
    "edge":         ["sharp", "blunt"],
    "surface":      ["brushed", "painted", "unpainted", "engraved", "rough", "smooth",
                     "shiny", "dull", "mossy"],
    "light":        ["bright", "dark", "clear"],
    "cleanliness":  ["clean", "dirty", "grimy", "muddy", "rusty"],
    "moisture":     ["wet", "dry", "damp"],
    "clarity":      ["cloudy", "foggy", "murky"],
    "cooking":      ["browned", "burnt", "caramelized", "cooked", "raw", "fresh",
                     "mashed", "pureed", "whipped"],
    "ripeness":     ["ripe", "unripe", "verdant", "wilted", "barren", "moldy"],
    "phase":        ["frozen", "melted", "molten", "thawed", "steaming", "runny", "viscous"],
    "cut":          ["cored", "cut", "diced", "peeled", "sliced"],
    "fill":         ["empty", "filled", "full", "deflated", "inflated", "spilled"],
    "openness":     ["open", "closed", "loose", "tight"],
    "orientation":  ["fallen", "standing", "toppled", "upright"],
    "weather_env":  ["sunny", "windblown", "cluttered", "pressed"],
}

# ---------------------------------------------------------------------------
# Object super-categories.
# ---------------------------------------------------------------------------
OBJ_SUPCATS = {
    "fruit":        ["apple", "banana", "berry", "fig", "lemon", "orange", "pear",
                     "persimmon", "tomato", "fruit"],
    "vegetable":    ["bean", "garlic", "potato", "nut", "vegetable"],
    "prepared_food":["bread", "cake", "cookie", "candy", "chocolate", "pasta", "pie",
                     "pizza", "sandwich", "soup", "sauce", "paste", "butter", "cheese",
                     "eggs", "salad"],
    "meat":         ["meat", "beef", "chicken", "salmon", "seafood"],
    "drink":        ["coffee", "milk", "tea", "water", "oil", "sugar"],
    "animal":       ["animal", "bear", "cat", "dog", "elephant", "horse", "tiger",
                     "snake", "iguana", "fish"],
    "metal":        ["aluminum", "brass", "bronze", "copper", "lead", "steel", "metal"],
    "mineral":      ["stone", "rock", "boulder", "granite", "clay", "coal", "diamond",
                     "gemstone", "sand", "dirt", "mud", "dust"],
    "fabric":       ["cotton", "silk", "wool", "velvet", "fabric", "thread", "ribbon",
                     "rope", "cord", "cable", "wire", "chains"],
    "material":     ["plastic", "rubber", "glass", "ceramic", "foam", "wax", "paint",
                     "concrete", "wood"],
    "building":     ["building", "castle", "church", "house", "tower", "library", "bridge",
                     "gate", "fence", "wall", "column", "roof", "ceiling", "floor", "deck",
                     "steps", "garage", "basement", "bathroom", "kitchen", "room", "shower",
                     "door", "window", "tile"],
    "place_nature": ["beach", "bay", "canyon", "cave", "cliff", "coast", "desert", "field",
                     "forest", "jungle", "island", "lake", "mountain", "ocean", "pond",
                     "pool", "river", "sea", "shore", "stream", "valley", "well", "trail",
                     "road", "highway", "street", "farm", "garden", "city", "town", "ground",
                     "creek"],
    "plant":        ["branch", "bush", "flower", "leaf", "moss", "palm", "plant", "redwood",
                     "roots", "tree", "tulip", "nest", "log"],
    "vehicle":      ["bike", "boat", "bus", "car", "truck"],
    "furniture":    ["bed", "cabinet", "chair", "desk", "table", "carpet", "mat", "frame",
                     "mirror", "furniture", "candle"],
    "clothing":     ["belt", "coat", "dress", "hat", "jacket", "pants", "shirt", "shoes",
                     "shorts", "tie", "clothes", "armor", "glasses"],
    "jewelry":      ["bracelet", "necklace", "ring", "jewelry", "coin", "penny", "key"],
    "tool_container":["blade", "knife", "sword", "screw", "gear", "handle", "hose", "tube",
                      "bucket", "basket", "bottle", "bowl", "box", "bag", "plate", "pot"],
    "electronics":  ["camera", "clock", "computer", "laptop", "keyboard", "phone", "fan",
                     "vacuum", "lightbulb", "drum", "wheel", "tire"],
    "elemental":    ["cloud", "sky", "smoke", "lightning", "fire", "flame", "ice", "snow",
                     "wave", "bubble", "balloon", "ball", "shell", "toy"],
    "document":     ["book", "card", "newspaper", "envelope", "paper"],
}


def _invert(groups, canonical, kind):
    name2id = {name: i for i, name in enumerate(groups.keys())}
    member2group = {}
    for gname, members in groups.items():
        for m in members:
            if m in member2group:
                raise RuntimeError(f"{kind}: '{m}' assigned to both "
                                   f"'{member2group[m]}' and '{gname}'")
            member2group[m] = gname
    # coverage check
    canon = set(canonical)
    assigned = set(member2group.keys())
    missing = canon - assigned
    extra = assigned - canon
    if missing:
        raise RuntimeError(f"{kind}: {len(missing)} unassigned -> {sorted(missing)}")
    if extra:
        raise RuntimeError(f"{kind}: {len(extra)} unknown names -> {sorted(extra)}")
    ids = [name2id[member2group[m]] for m in canonical]
    return list(groups.keys()), ids


def main():
    attr_group_names, attr_group_id = _invert(ATTR_GROUPS, ATTRS, "attr")
    obj_supcat_names, obj_supcat_id = _invert(OBJ_SUPCATS, OBJS, "obj")

    out = {
        "attr_index": ATTR_INDEX,
        "obj_index": OBJ_INDEX,
        "attr_group_names": attr_group_names,
        "obj_supcat_names": obj_supcat_names,
        "attr_group_id": attr_group_id,
        "obj_supcat_id": obj_supcat_id,
    }
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
    print(f"wrote {OUT}")
    print(f"  attrs: {len(ATTRS)} -> {len(attr_group_names)} groups")
    print(f"  objs : {len(OBJS)} -> {len(obj_supcat_names)} supcats")
    # group-size sanity
    from collections import Counter
    ac = Counter(attr_group_id); oc = Counter(obj_supcat_id)
    print("  attr group sizes:", {attr_group_names[k]: v for k, v in sorted(ac.items())})
    print("  obj supcat sizes:", {obj_supcat_names[k]: v for k, v in sorted(oc.items())})


if __name__ == "__main__":
    main()
