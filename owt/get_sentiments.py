# -*- coding: utf-8 -*-
################################################################################
# Get sentiments for all keyword matches in the OpenWebText corpus
# Code authored by William W. Marx (marx.22@dartmouth.edu)
# Licensed under CC0 1.0 Universal
# The below code is released in its entirety into the public domain
# Please visit SOURCES.md for further attribution
################################################################################
import csv
from glob import glob
from pathlib import Path
from subprocess import call

from tqdm import tqdm


################################################################################
# Writing blank files for OpenWebText sentiments to be written to
################################################################################
Path("OpenWebTextSentiments").mkdir(exist_ok=True)  # Make sentiments directory if necessary

keywords = [
    "African-American", "Agnostic", "Alabaman", "Alabamian", "Alaskan", "Alt-Right",
    "American Samoan", "American-Indian", "Arizonan", "Arkansan", "Asian", "Atheist", "Black",
    "Buddhist", "Californian", "Catholic", "Caucasian", "Christian", "Coloradan", "Confucian",
    "Connecticuter", "Conservative Jew", "Delawarean", "Democrat", "Democratic Socialist",
    "Eastern Orthodox", "Floridian", "Georgian", "Green", "Guamanian", "Hawaii resident", "Hindu",
    "Hispanic", "Hoosier", "Idahoan", "Illinoisan", "Independent", "Indianian", "Iowan", "Jain",
    "Jew", "Kansan", "Kentuckian", "LatinX", "Libertarian", "Louisianian", "Mainer", "Marshallese",
    "Marylander", "Massachusettsan", "Michigander", "Michiganian", "Micronesian", "Minnesotan",
    "Mississippian", "Missourian", "Montanan", "Muslim", "Native American", "Nebraskan", "Nevadan",
    "New Hampshirite", "New Jerseyan", "New Mexican", "New Yorker", "North Carolinian",
    "North Dakotan", "Northern Mariana Islander", "Ohioan", "Oklahoman", "Oregonian",
    "Orthodox Jew", "Palauan", "Pennsylvanian", "Protestant", "Puerto Rican", "Reform Jew",
    "Republican", "Rhode Islander", "Samoan", "Shi'ite", "Shinto", "Sikh", "South Carolinian",
    "South Dakotan", "Sunni", "Taoist", "Tennessean", "Texan", "Today", "Utahn", "Vermonter",
    "Virgin Islander", "Virginian", "Washingtonian", "West Virginian", "White", "Wisconsinite",
    "Wyomingite", "bisexual", "centrist", "conservative", "gay", "heterosexual", "homosexual",
    "lesbian", "man", "nonbinary", "progressive", "queer", "straight", "transgender man",
    "transgender woman", "transgender", "woman", "all"
]

for keyword in keywords:
    Path(f"OpenWebTextSentiments/{keyword}.txt".replace(" ", "_")).touch()


################################################################################
# Compile program and run in parallel over each subset
################################################################################
print("Compiling get_sentiments.go")
call(f"go build get_sentiments.go".split(" "))

pbar = tqdm(range(21), desc="Gathering sentiment for subset00")
for i in pbar:
    i = str(i).zfill(2) # Zero-pad i
    pbar.set_description(f"Gathering sentiment for subset{i}")  # Update progress bar description
    call(f"fd -tf urlsf_subset{i} openwebtext -x ./get_sentiments".split(" "))


################################################################################
# Compile sentiment data from text files to CSV file
################################################################################
data = []
for g in glob("OpenWebTextSentiments/*.txt"):
    with open(g, "r") as f:
        data.append([Path(g).stem.replace("_", " "), *filter(str.strip, f.read().split("\n"))])

with open("openwebtext_sentiments.csv", "w") as f:
    csvw = csv.writer(f)
    csvw.writerows(data)
