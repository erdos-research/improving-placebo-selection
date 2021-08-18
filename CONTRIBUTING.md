# Contributing to GPT-2 Generates Biased Texts

## Reporting an issue
Perhaps the easiest way to improve this project is quite simply to report any issues you run into
along the way. Please try to describe the issue you're having as precisely as is helpful. If you
don't receive a prompt response, feel free to email <null@marx.studio> and follow up.

## Improving our code
If you'd like to improve our code in any way, we'd be quite receptive — just submit a pull request.
We seek to make this code as efficient and broadly accessible as possible. While Python certainly
isn't known for its efficiency, its easy to read, a trade off we find more than fulfilling.
Otherwise, any changes are fair game. See below the top improvements we want to tackle:
- Automated native Windows installation in [`install.sh`](./install.sh)
- arm64 port of [tensorflow/tensorflow:1.15.2-gpu](https://hub.docker.com/layers/tensorflow/tensorflow/1.15.2-gpu/images/sha256-da7b6c8a63bdafa77864e7e874664acfe939fdc140cb99940610c34b8c461cd0)
for reproducible GPT-2 Placebo generation on Apple Silicon
