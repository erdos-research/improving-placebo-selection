# -*- coding: utf-8 -*-
################################################################################
# Download OpenWebText corpus and create text files to write sentiment to
# Code authored by William W. Marx (marx.22@dartmouth.edu)
# Licensed under CC0 1.0 Universal
# The below code is released in its entirety into the public domain
# Please visit SOURCES.md for further attribution
################################################################################
from pathlib import Path
from subprocess import call

import gdown


################################################################################
# Download OpenWebText corpus
################################################################################
# If OpenWebText corpus not already downloaded, download it
if not Path("openwebtext.tar.xz").exists() or Path("openwebtext").is_dir():
    print("Attempting to download OpenWebText (12GB tar.xz)")
    gdown.download("https://drive.google.com/uc?id=1EA5V0oetDCOke7afsktL_JDQ-ETtNOvx",
                   "openwebtext.tar.xz",
                   quiet=False)

    # Check to see if download was successful, if not, require user to install manually.
    if not Path("openwebtext.tar.xz").exists():
        raise Exception("\n".join([
            "\033[31mOpenWebText download unsuccessful.\033[0m",
            "Please visit https://skylion007.github.io/OpenWebTextCorpus/ and download the corpus into this directory (`owt`)."
        ]))
else:
    print("OpenWebText already exists, moving on")


################################################################################
# Decompress corpus
################################################################################
if not Path("openwebtext").is_dir():
    print("Decompressing openwebtext.tar.xz")
    call("tar -xzf openwebtext.tar.xz".split(" "))

    print("Decompressing individual OpenWebText files")
    call("fd -e xz -x unxz".split(" "))
else:
    print("Nothing to do, data already decompressed")
