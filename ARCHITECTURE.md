# Project Architecture
While this project is quite simple in architecture, organization is as follows:
- Main tasks delegated to `Makefile`
- Main requirements listed in README
- Python package requirements in `requirements.txt`
- Main project code exists in the [`src/`](src/) directory
	+ [`generate.py`](./generate.py) is used to generate new placebos
	+ [`compile.py`](./compile.py) compiles all text files of generated placebos into a main CSV file
	+ [`analyze.py`](./analyze.py) computes sentiments and substantive content for all placebos
	+ [`visualize.py`](./visualize.py) generates figures used throughout our letter
	+ [`owt`](./owt) directory containing code to analyze [OpenWebText](https://skylion007.github.io/OpenWebTextCorpus/)
		* [`download.py`](./owt/download.py) — download and decompress the OpenWebText corpus
		* [`get_sentiments.py`](./owt/get_sentiments.py) — get sentiments for all documents and matches
in the OpenWebText corpus and save as a CSV file
		* [`get_sentiments.go`](./owt/get_sentiments.go) — quickly get sentiments for all documents in a
single file (this operation is handled entirely by
[`get_sentiments.py`](./owt/get_sentiments.py))
- Data for reproduction exists in the [`data`](./data) directory
	- `gpt2_generated_placebos0*` are the compressed raw placebo text we generated
	- `openwebtext_sentiments0*` is our compressed OpenWebText sentiment analysis CSV file
	- [`seed_phrases.txt`](./data/seed_phrases.txt) contains all seed phrases we used
	- [`group_identifiers.json`](./data/group_identifiers.json) organizes group identifiers by their respective categories
