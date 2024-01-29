# -*- coding: utf-8 -*-
################################################################################
# Generate placebos with GPT-2 and GPT-3.5-Turbo
# Code authored by William W. Marx (marx.22@dartmouth.edu)
# Code adopted from Porter and Velez (https://bit.ly/placebo_tools)
# Licensed under CC0 1.0 Universal
# The below code is released in its entirety into the public domain
# Please visit SOURCES.md for further attribution
################################################################################
import sys
from pathlib import Path
from subprocess import call

# Get current working directory
cwd = Path(__file__).parent.resolve()

try:
	import gpt_2_simple as gpt2
	from tqdm import tqdm
	from openai import OpenAI
except ModuleNotFoundError:
	print("\033[31mMissing necessary requirements.\033[0m")
	# Ask user if they want to install missing packages and if so, do it
	generate_requirements_fp = (cwd / "requirements.generate.txt").resolve()
	if input("Install `gpt-2-simple`, `tqdm`, `openai` now? [Y/n]: ").strip().lower() != "n":
		call(f"python3 -m pip install -U -r {generate_requirements_fp}".split(" "))
	else:
		raise ModuleNotFoundError("Must install `gpt_2_simple` and `tqdm`")

if len(sys.argv) != 2:  # Inappropriate number of arguments given
	raise Exception("USAGE: python generate_placebos.py [SEED_PHRASE_FILE]")

if not Path(sys.argv[1]).exists():  # Passed file does not exist
	raise Exception(f"{sys.argv[1]} does not exist.")

################################################################################
# GPT-2 Placebo Generation
################################################################################
print("====== Generating Placebos with GPT-2 ======")
sess = gpt2.start_tf_sess()  # Start session
print("Tensorflow session started")

if not Path("models/1558M").is_dir():
	print("1558M model not found, downloading now.")
	gpt2.download_gpt2(model_name="1558M")  # Download model if necessary

gpt2.load_gpt2(sess, model_name="1558M")  # Load model
print("1558M model loaded")

with open(sys.argv[1], "r") as f:
	seed_phrases = list(set(filter(None, f.read().split("\n"))))  # Read seed phrases from file
print("Seed phrases read into memory")

# Make gpt2_generated_placebos directory if necessary
gpt2_placebos_dir = (cwd / "../data/gpt2_generated_placebos").resolve()
gpt2_placebos_dir.mkdir(parents=True, exist_ok=True)

seed_phrase_pbar = tqdm(sorted(seed_phrases), desc=seed_phrases[0])
for sp in seed_phrase_pbar:
	seed_phrase_pbar.set_description(sp)  # Update progress bar description
	gpt2.generate_to_file(sess,
	                      destination_path=gpt2_placebos_dir / f"{sp.replace(' ', '_')}.txt",
	                      model_name="1558M",
	                      prefix=sp,
	                      length=200,
	                      temperature=0.7,
	                      nsamples=250,
	                      batch_size=25)

################################################################################
# GPT-3.5 Turbo Placebo Generation
################################################################################
print("====== Generating Placebos with GPT-3.5 ======")
gpt3_5_placebos_dir = (cwd / "../data/gpt3_5_generated_placebos").resolve()
gpt3_5_placebos_dir.mkdir(parents=True, exist_ok=True)
client = OpenAI()

seed_phrase_pbar = tqdm(sorted(seed_phrases), desc=seed_phrases[0])
for sp in seed_phrase_pbar:
	seed_phrase_pbar.set_description(sp)  # Update progress bar description
	response = client.chat.completions.create(model="gpt-3.5-turbo-1106",
	                                          messages=[{
	                                              "role": "user",
	                                              "content": sp
	                                          }],
	                                          max_tokens=200,
	                                          temperature=0.7,
	                                          n=100)

	fp = gpt3_5_placebos_dir / f"{sp.replace(' ', '_')}.txt" if sp != "Today, " else gpt3_5_placebos_dir / "Today.txt"
	with open(gpt3_5_placebos_dir / f"{sp.replace(' ', '_')}.txt", "w") as f:
		separator = "\n" + "=" * 20 + "\n"
		placebos = [sp + choice.message.content for choice in response.choices]
		f.write(separator.join(placebos) + "\n")
