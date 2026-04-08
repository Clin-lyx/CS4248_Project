from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from systems.system_a.template_utils import preserves_anchors
from systems.system_b_utils import normalized_token_edit_distance, semantic_similarity_score


UPDATES = {
    "sar_000157": "arizona iced tea unveils oversized 4-foot-tall cans",
    "sar_000180": "man feels grateful to see another customer wearing jeans in nice restaurant",
    "sar_000231": "fire department deploys trucks without markings",
    "sar_000264": "senate panel unanimously approves chris wray's nomination as fbi director, because suspense clearly overstayed its welcome",
    "sar_000369": "boss arrives at work dressed like the man who fires sean",
    "sar_000377": "mother brings 4 more shirt options to son in gap dressing room",
    "sar_000403": "ad campaign appeals to basic human intelligence",
    "sar_000466": "two hipsters angrily accuse each other of being hipsters",
    "sar_000483": "amid an industry boom, incarceration for weed still threatens black women, because prosperity always knows who to skip",
    "sar_000494": "america's fascination with ally mcbeal ends badly",
    "sar_000603": "cactus scientists advise drinking 8 cups of water per year",
    "sar_001367": "miss america pageant adds sweatpants and messy bun contest",
    "sar_001431": "farmer chases a fifth wedding party out of barn this month",
    "sar_001449": "kroger recalls 35,000 pounds of ground beef over possible contamination",
    "sar_001532": "hank williams jr. honored by institute for football preparedness",
    "sar_001595": "television programming dominated by surreal reality content",
    "sar_001602": "newlywed britney spears hangs bloody sheet in window for reporters",
    "sar_001610": "burger king unveils new low-fat cashier concept",
    "sar_001614": "pope francis crushes a small demon on papal apartment floor",
    "sar_001616": "oxiclean introduces stain-removing fabric scissors",
    "sar_001649": "study says controlled washington, d.c. wildfires are considered crucial for restoring political environment",
    "sar_001688": "jay leno reconsiders retirement after georgia woman sets boyfriend's crotch on fire",
    "sar_001692": "right-to-kill advocate opposes right-to-die policy",
    "sar_001701": "pope wins wafer-eating contest",
    "sar_001704": "study says college remains more worthwhile than spending 4 years chained to a radiator",
    "sar_001737": "vagina develops five o'clock shadow",
    "sar_001742": "parody movie script is one crotch-hitting joke away from approval",
    "sar_001747": "cautious senior locks screen door",
    "sar_001760": "nation breathes a sigh of lingering unease",
    "sar_001794": "vince gilligan's brain spoils the final season of 'breaking bad' for its creator",
    "sar_001822": "neurosurgeon is heckled from observation deck",
    "sar_001830": "clinton becomes first president to clear 18 feet in pole vault",
    "sar_001839": "keystone veto buys the environment at least 3 or 4 more hours",
    "sar_001883": "bernie sanders slams trump: 'that kind of crap is not going to work in the united states,' because apparently the point still needed repeating",
    "sar_002095": "donald glover did what any fan would do after being cast in 'solo,' because subtle excitement is for weaker nerds",
    "sar_002109": "deadly california wildfire near big sur set to explode in size, because nature apparently wanted a sequel",
    "sar_002153": "police near st. louis quash peaceful protest by declaring it an unlawful assembly, because peace keeps making authority nervous",
    "sar_002215": "mom has a full dinner party roster",
    "sar_002242": "exercise is briefly considered",
    "sar_002253": "kevin spacey responds to assault allegations by seeking treatment for homosexuality",
    "sar_002296": "women's prison riot seems gratuitous",
    "sar_002304": "parents mark child's width on kitchen wall",
    "sar_002305": "bangladesh runs out of available people",
    "sar_002337": "head-on collision with a ford, because impact needed a sponsor",
    "sar_002411": "ice detains tim kaine over speaking spanish at campaign rally",
    "sar_002422": "that knife guy from high school is arrested in knife-related incident",
    "sar_002424": "scientists warn americans to stay away from a bird",
    "sar_002471": "'new york post' publishes report alleging alexandria ocasio-cortez has a 9-figure social security number",
    "sar_002508": "kfc releases a family-size nugget",
    "sar_002534": "texas governor legalizes formerly banned wrestling move",
    "sar_002596": "margaret atwood: 'the handmaids are meant to be aliens'",
    "sar_002617": "indonesian mother sews halloween costumes for 60,000 children",
    "sar_002717": "clinton campaign requests cnn stock dressing room with 4 pounds of flavorless protein paste",
    "sar_002782": "busybody fireman ruins an attempted suicide",
    "sar_002816": "washing machine causes man to lose trust",
    "sar_002879": "u.s. middlemen seek protection from elimination",
    "sar_002903": "christian pornographer declines to film sex tape for gay couple",
    "sar_002929": "man nods at mechanic as if he understands",
    "sar_002963": "gop follows through on 2009 promise to block president's healthcare bill",
}


def main() -> None:
    path = PROJECT_ROOT / "artifacts" / "system_b" / "pseudo_pairs_filtered.jsonl"
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["id"] in UPDATES:
                row["target_text"] = UPDATES[row["id"]]
                row["semantic_similarity"] = semantic_similarity_score(row["source_text"], row["target_text"])
                row["edit_ratio"] = normalized_token_edit_distance(row["source_text"], row["target_text"])
                row["anchors_preserved"] = preserves_anchors(row["anchors"], row["target_text"])
            rows.append(row)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("updated", len(UPDATES))


if __name__ == "__main__":
    main()
