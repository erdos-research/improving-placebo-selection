# -*- coding: utf-8 -*-
################################################################################
# Compile generated placebos into single CSV file
# Code authored by William W. Marx (marx.22@dartmouth.edu)
# Licensed under CC0 1.0 Universal
# The below code is released in its entirety into the public domain
# Please visit SOURCES.md for further attribution
################################################################################
import csv
import json
import re
from glob import glob
from pathlib import Path
from itertools import chain

# Load group indentifiers from json file
with open("data/group_identifiers.json", "r") as f:
    group_ids = json.load(f)

# Separate demographic identifiers from names
demographic_ids = list(chain.from_iterable([v for k, v in group_ids.items() if k != "Name"]))
names = list(chain.from_iterable(group_ids["Name"].values()))

# Create dict to handle different P.O.S. (e.g. Jewish -> Jew)
dem_id_dict = {k: k for k in filter(lambda x: type(x) is str, demographic_ids)}
for k in filter(lambda x: type(x) is list, demographic_ids):
    for v in k:
        dem_id_dict[v] = k[0]


def get_ids(demographic_id, seed_phrase):
    """Get identifiers from a seed phrase, return None on no match"""
    # Make all demographic id's a list for convenience
    if type(demographic_id) is str:
        demographic_id = [demographic_id]
    base_id = demographic_id[0]  # Handle different P.O.S. (e.g. Jewish -> Jew)
    for d in demographic_id:
        if re.search(f"^{d}$|^{d} .+", seed_phrase):
            # Demographic ID occurs at start of seed phrase
            return base_id, re.sub(f"^{d}", "", seed_phrase).strip()
        elif re.search(f".+ {d}$", seed_phrase):
            # Demographic ID occurs at end of seed phrase
            return re.sub(f"{d}$", "", seed_phrase).strip(), base_id
    return "", ""


def split_seed_phrase(seed_phrase):
    """Split a seed phrase into its component identifiers."""
    # Get rid of negibile words "person" and "supporter"
    seed_phrase = re.sub(r"person|supporter|Party", "", seed_phrase).strip()

    # If seed phrase is a name or "Today", return it
    if seed_phrase in [*names, "Today"]:
        return seed_phrase, ""

    # Check for demographic identifiers
    for demographic_id in demographic_ids:
        id0, id1 = get_ids(demographic_id, seed_phrase)
        if id0 == "transgender" and id1 in ["man", "woman"]:
            # Transgender man and woman are ids of their own, don't allow duplicates
            return " ".join([id0, id1]), ""
        if id0 in dem_id_dict.keys() and (id1 in dem_id_dict.keys() or not id1):
            return dem_id_dict.get(id0, ""), dem_id_dict.get(id1, "")

    # Return nothing if nothing found
    return "", ""


# Check if data/gpt2_generated_placebos direcroty exists before proceding
if Path("data/gpt2_generated_placebos").is_dir():
    seed_re = re.compile(r"^Today,_(an?_)?")  # Regex to get seed phrase
    splitter = re.compile(r"={20}")  # Regex to split gpt-2-simple

    # Parse through raw data files
    csv_rows = []
    for fp in glob("data/gpt2_generated_placebos/*.txt"):
        # Get seed phrase from filename
        seed_phrase = seed_re.sub("", Path(fp).stem).replace("_", " ")
        id0, id1 = split_seed_phrase(seed_phrase)  # Get identifiers from seed phrase
        with open(fp, "r") as f:
            # Split text into placebos
            placebos = filter(None, map(str.strip, splitter.split(f.read())))
            csv_rows.append([id0, id1, *placebos])

    if not Path("results").is_dir():
        Path("results").mkdir()  # Make results directory if necessary

    # Write placebos to CSV file
    with open("results/GPT2_generated_placebos.csv", "w") as f:
        csvw = csv.writer(f)
        csvw.writerow(["Identifier0", "Identifier1", *[f"Placebo{i}" for i in range(250)]])
        csvw.writerows(sorted(csv_rows, key=lambda x: " ".join(x[:2]).lower()))
else:
    # If no raw data, instruct user on what to do
    print("\033[31mDirectory 'data/gpt2_generated_placebos' not found.\033[0m")
    print(
        "Either decompress results (`make decompress-data`) or generate new results (`make placebos`)."
    )
