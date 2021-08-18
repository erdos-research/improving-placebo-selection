# GPT-2 Generates Biased Texts
[![License: CC0-1.0](https://img.shields.io/badge/License-CC0%201.0-lightgrey.svg)](http://creativecommons.org/publicdomain/zero/1.0/)
[![OmniOpen Platinum](https://github.com/concurrent-studio/OmniOpen/raw/master/badges/platinum.svg)](https://concurrent.studio/omniopen#platinum)
[![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](TODO)  
Code accompanying the paper [GPT-2 Generates Biased Texts]() by
[Charles Crabtree](https://charlescrabtree.com) and [William W. Marx](https://marx.design).

*ABSTRACT GOES HERE*

## Install
To install all necessary requirements, run `bash install.sh`. We recommend using Docker, but native
installs for macOS and GNU/Linux are similarly available through the interactive install script.

If not running via Docker, the following items will be installed if not already extant:
- [python 3.6](https://www.python.org)
- [go](https://golang.org)
- [7zip](https://www.7-zip.org)
- [fd](https://github.com/sharkdp/fd)

## Usage
### Reproducing our results
To reproduce our results including CSV file with all generated placebos, CSV file with all placebo
sentiments, CSV analysis file with basic stats and all graphs, simply run `make reproduce`.

### Generating new results
To reproduce all results as described above with the same seed phrases we used, run `make all`.

To generate results with a different set of seed phrases, run
`python3 generate.py [SEED PHRASE FILE]`. You will then have to modify [`compile.py`](./compile.py),
[`analyze.py`](./analyze.py) and [`visualize.py`](./visualize.py), as they rely on the
[`data/group_identifiers.json`](data/group_identifiers.json) file which is tailored to the seed
phrases we used.

### Reproducing OpenWebText results
To reproduce our OpenWebText sentiment analysis findings, run `make openwebtext`.

## Project Architecture
See [`ARCHITECTURE.md`](./ARCHITECTURE.md)

## Code of Conduct
See [`CODE_OF_CONDUCT.md`](./CODE_OF_CONDUCT.md)

## Contributing
See [`CONTRIBUTING.md`](./CONTRIBUTING.md)
