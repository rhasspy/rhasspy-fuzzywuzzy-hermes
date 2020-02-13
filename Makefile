SHELL := bash
PYTHON_NAME = rhasspyfuzzywuzzy_hermes
PACKAGE_NAME = rhasspy-fuzzywuzzy-hermes
SOURCE = $(PYTHON_NAME)
PYTHON_FILES = $(SOURCE)/*.py tests/*.py *.py
SHELL_FILES = bin/* debian/bin/*
PIP_INSTALL ?= install
DOWNLOAD_DIR = download

.PHONY: reformat check dist venv test pyinstaller debian deploy rhasspy-libs

version := $(shell cat VERSION)
architecture := $(shell bash architecture.sh)

# -----------------------------------------------------------------------------
# Python
# -----------------------------------------------------------------------------

reformat:
	scripts/format-code.sh $(PYTHON_FILES)

check:
	scripts/check-code.sh $(PYTHON_FILES)

venv: rhasspy-libs
	scripts/create-venv.sh

test:
	echo "No tests yet"

coverage:
	coverage report -m

dist: sdist debian

sdist:
	python3 setup.py sdist

# -----------------------------------------------------------------------------
# Docker
# -----------------------------------------------------------------------------

docker: pyinstaller
	docker build . -t "rhasspy/$(PACKAGE_NAME):$(version)" -t "rhasspy/$(PACKAGE_NAME):latest"

deploy:
	echo "$$DOCKER_PASSWORD" | docker login -u "$$DOCKER_USERNAME" --password-stdin
	docker push "rhasspy/$(PACKAGE_NAME):$(version)"

# -----------------------------------------------------------------------------
# Debian
# -----------------------------------------------------------------------------

pyinstaller:
	scripts/build-pyinstaller.sh "${architecture}" "${version}"

debian:
	scripts/build-debian.sh "${architecture}" "${version}"

# -----------------------------------------------------------------------------
# Downloads
# -----------------------------------------------------------------------------

# Rhasspy development dependencies
rhasspy-libs: $(DOWNLOAD_DIR)/rhasspy-fuzzywuzzy-0.1.1.tar.gz $(DOWNLOAD_DIR)/rhasspy-hermes-0.1.6.tar.gz

$(DOWNLOAD_DIR)/rhasspy-fuzzywuzzy-0.1.1.tar.gz:
	mkdir -p "$(DOWNLOAD_DIR)"
	curl -sSfL -o $@ "https://github.com/rhasspy/rhasspy-fuzzywuzzy/archive/master.tar.gz"

$(DOWNLOAD_DIR)/rhasspy-hermes-0.1.6.tar.gz:
	mkdir -p "$(DOWNLOAD_DIR)"
	curl -sSfL -o $@ "https://github.com/rhasspy/rhasspy-hermes/archive/master.tar.gz"
