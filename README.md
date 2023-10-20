# Improving Placebo Selection in Survey Experiments
[![License: CC0-1.0](https://img.shields.io/badge/License-CC0%201.0-lightgrey.svg)](http://creativecommons.org/publicdomain/zero/1.0/)
[![OmniOpen Platinum](https://github.com/concurrent-studio/OmniOpen/raw/master/badges/platinum.svg)](https://concurrent.studio/omniopen#platinum)  
Code accompanying the paper [Improving Placebo Selection in Survey Experiments]() by
[Charles Crabtree](https://charlescrabtree.com) and [William W. Marx](https://marx.sh).

How should researchers create placebo conditions in survey experiments? Porter and Velez (2021) recommend that researchers use automated processes to construct a large corpus of placebo conditions, which are then randomly assigned to participants in the placebo group during survey implementation. Based on our empirical work, we suggest that researchers use caution if they employ the tool recommended for placebo construction, OpenAI’s semi-supervised language model GPT-2. We conduct the most extensive assessment of GPT-2’s biases by measuring the sentiment of 1,083,750 potential placebos generated across 4,335 unique seed phrases. We show that the polarities of placebos vary tremendously across seed phrases, depending on the race/ethnicity, gender, sexual and religious orientation, political affiliation, political ideology, or state or territory demonym mentioned in them. We also show considerable heterogeneity in the substance of placebos across seed phrases. Comparing our results to a similar analysis of the text corpus used to train GPT-2, we find that the language model not only learns to reproduce biases in source material but also magnifies them. To deal with these issues, we provide a tool to mitigate the effects of these biases and provide best practice recommendations for automatic placebo generation.

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
