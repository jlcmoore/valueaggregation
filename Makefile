PUBLIC_EXPORT_DEST ?= /private/tmp/moral-parliament-public
PYTHON_VENV := env-aggregation/bin/python
NPM_INSTALL_CMD := if [ -f package-lock.json ]; then npm ci; else npm install; fi

fmt: pyfmt webfmt

pyfmt:
	git ls-files -z '*.py' '*.ipynb' | \
		xargs -0 $(PYTHON_VENV) -m isort --profile black
	git ls-files -z '*.py' '*.ipynb' | \
		xargs -0 $(PYTHON_VENV) -m black

webfmt:
	@if [ ! -x ./node_modules/.bin/prettier ]; then \
		$(NPM_INSTALL_CMD); \
	fi
	git ls-files -z '*.html' '*.css' '*.js' | \
		xargs -0 ./node_modules/.bin/prettier --write

test:
	cd src/ && ../env-aggregation/bin/python -m unittest tests

init:
	python3.11 -m venv env-aggregation -
	env-aggregation/bin/pip install --editable .
	env-aggregation/bin/pip install -r requirements-dev.txt
	env-aggregation/bin/python -m ipykernel install --user --name "env-aggregation"

data:
	mkdir external_data

	mkdir external_data/nlpositionality
	# NLPositionality Data
	# More info here: https://github.com/liang-jenny/nlpositionality
	curl "https://delphi-litw.apps.allenai.org/api/v1/dataset?type=raw" \
		> external_data/nlpositionality/social-acceptability_raw.csv
	curl "https://delphi-litw.apps.allenai.org/api/v1/dataset?type=processed" \
		> external_data/nlpositionality/social-acceptability_processed.csv
	curl "https://toxicity-litw.apps.allenai.org/api/v1/dataset?type=raw" \
		> external_data/nlpositionality/toxicity_raw.csv
	curl "https://toxicity-litw.apps.allenai.org/api/v1/dataset?type=processed" \
		> external_data/nlpositionality/toxicity_processed.csv

	curl "https://population.un.org/wpp/Download/Files/1_Indicators%20(Standard)/CSV_FILES/WPP2022_Demographic_Indicators_Medium.zip" \
		> external_data/un_population.zip
	cd external_data; unzip un_population.zip

	mkdir external_data/moral_machines
	# Moral Machines Data
	# Look at their data page, here:
	# 	https://osf.io/3hvt2/
	# All datasets as zip
	# https://files.osf.io/v1/resources/3hvt2/providers/osfstorage/5b53cdbc69e43a000fd3a035/?zip=
	# All code as zip
	# https://files.osf.io/v1/resources/3hvt2/providers/osfstorage/5b53cdbc69e43a000fd3a035/?zip=

	# This is 3 GB
	wget https://osf.io/download/tdqrn/ \
		--output-document=external_data/moral_machines/SharedResponses.csv.tar.gz
	tar -xzf external_data/moral_machines/SharedResponses.csv.tar.gz --directory external_data/moral_machines/
	rm external_data/moral_machines/SharedResponses.csv.tar.gz

	wget https://osf.io/download/q5dk4/ \
		--output-document=external_data/moral_machines/CountryChangePr.csv.tar.gz
	tar -xzf external_data/moral_machines/CountryChangePr.csv.tar.gz --directory external_data/moral_machines/	
	rm external_data/moral_machines/CountryChangePr.csv.tar.gz

	# This is 1 GB
	wget https://osf.io/download/ukmzd/ \
		--output-document=external_data/moral_machines/SharedResponsesFirstFullSessions.csv.tar.gz
	tar -xzf external_data/moral_machines/SharedResponsesFirstFullSessions.csv.tar.gz \
		--directory external_data/moral_machines/
	rm external_data/moral_machines/SharedResponsesFirstFullSessions.csv.tar.gz

public-export:
	bash scripts/create_public_repo_copy.sh "$(PUBLIC_EXPORT_DEST)"

.PHONY: data fmt init public-export pyfmt test webfmt
