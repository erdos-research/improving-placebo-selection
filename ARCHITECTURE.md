# Project Architecture
While this project is quite simple in architecture, organization is as follows:
- Main tasks delegated to `Makefile`
- Install script for project requirements in [`install.sh`](./install.sh)
- Python package requirements in `requirements.*.txt` with each containing only packages necessary
for a given script
for their individual scripts
- Main code in [`generate.py`](./generate.py), [`compile.py`](./compile.py),
[`analyze.py`](./analyze.py) and [`visualize.py`](./visualize.py)
- [`data`](./data) directory
	- `gpt2_generated_placebos0*` are the compressed raw placebo text we generated
	- [`openwebtext_sentiments.7z`](./data/openwebtext_sentiments.7z) is our compressed OpenWebText sentiment analysis CSV file
	- [`seed_phrases.txt`](./data/seed_phrases.txt) contains all seed phrases we used
	- [`group_identifiers.json`](./data/group_identifiers.json) organizes group identifiers by their
respective categories
- [`owt`](./owt) directory containing code to analyze
[OpenWebText](https://skylion007.github.io/OpenWebTextCorpus/)
	- [`download.py`](./owt/download.py) — download and decompress the OpenWebText corpus
	- [`get_sentiments.py`](./owt/get_sentiments.py) — get sentiments for all documents and matches in
the OpenWebText corpus and save as a CSV file
	- [`get_sentiments.go`](./owt/get_sentiments.go) — quickly get sentiments for all documents in a
single file (this operation is handled entirely by [`get_sentiments.py`](./owt/get_sentiments.py))
