# -*- coding: utf-8 -*-
################################################################################
# Generate placebos with GPT-2
# Code authored by William W. Marx (marx.22@dartmouth.edu)
# Code adopted from Porter and Velez (https://bit.ly/placebo_tools)
# Licensed under CC0 1.0 Universal
# The below code is released in its entirety into the public domain
# Please visit SOURCES.md for further attribution
################################################################################
import sys
from pathlib import Path
from subprocess import call

try:
    import gpt_2_simple as gpt2
    from tqdm import tqdm
except ModuleNotFoundError:
    print("\033[31mMissing necessary requirements.\033[0m")
    # Ask user if they want to install missing packages and if so, do it
    if input("Install `gpt_2_simple` and `tqdm` now? [Y/n]: ").strip().lower() != "n":
        call("python3 -m pip install -U -r requirements.generate.txt".split(" "))
    else:
        raise ModuleNotFoundError("Must install `gpt_2_simple` and `tqdm`")

if len(sys.argv) != 2:  # Inappropriate number of arguments given
    raise Exception("USAGE: python generate_placebos.py [SEED_PHRASE_FILE]")

if not Path(sys.argv[1]).exists():  # Passed file does not exist
    raise Exception(f"{sys.argv[1]} does not exist.")

sess = gpt2.start_tf_sess()  # Start session
print("Tensorflow session started")

if not Path("models/1558M").is_dir():
    print("1558M model not found, downloading now.")
    gpt2.download_gpt2(model_name="1558M")  # Download model if necessary

gpt2.load_gpt2(sess, model_name="1558M")  # Load model
print("1558M model loaded")

with open(sys.argv[1], "r") as f:
    seed_phrases = set(filter(None, f.read().split("\n")))  # Read seed phrases from file
print("Seed phrases read into memory")

# Make gpt2_generated_placebos directory if necessary
Path("data/gpt2_generated_placebos").mkdir(parents=True, exist_ok=True)

seed_phrase_pbar = tqdm(sorted(seed_phrases), desc=seed_phrases[0])
for sp in seed_phrase_pbar:
    seed_phrase_pbar.set_description(sp)  # Update progress bar description
    gpt2.generate_to_file(sess,
                          destination_path=f"data/gpt2_generated_placebos/{sp.replace(' ', '_')}",
                          model_name="1558M",
                          prefix=sp,
                          length=200,
                          temperature=0.7,
                          nsamples=250,
                          batch_size=25)
