.PHONY: decompress openwebtext reproduce all clean

decompress:
	@cat data/gpt2_generated_placebos0* > data/gpt2_generated_placebos.7z && \
		7z x data/gpt2_generated_placebos.7z -odata
	@cat data/openwebtext_sentiments0* > data/openwebtext_sentiments.7z && \
		7z x data/openwebtext_sentiments.7z -odata

openwebtext:
	@cd owt
	@python3 download.py
	@python3 get_sentiments.py
	@cd ..

reproduce:
	@make decompress
	@python3 compile.py
	@python3 analyze.py
	@python3 visualize.py

all:
	@python3 generate.py data/seed_phrases.txt
	@python3 compile.py
	@python3 analyze.py
	@python3 visualize.py

clean:
	@rm -f data/gpt2_generated_placebos.7z
	@rm -f data/openwebtext_sentiments.csv
	@rm -f owt/get_sentiments
	@rm -rf data/gpt2_generated_placebos
	@rm -rf results
	@rm -rf owt/OpenWebTextSentiments/
