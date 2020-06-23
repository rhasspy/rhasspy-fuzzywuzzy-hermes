# Rhasspy Fuzzywuzzy Hermes

[![Continous Integration](https://github.com/rhasspy/rhasspy-fuzzywuzzy-hermes/workflows/Tests/badge.svg)](https://github.com/rhasspy/rhasspy-fuzzywuzzy-hermes/actions)
[![GitHub license](https://img.shields.io/github/license/rhasspy/rhasspy-fuzzywuzzy-hermes.svg)](https://github.com/rhasspy/rhasspy-fuzzywuzzy-hermes/blob/master/LICENSE)

Implements `hermes/nlu` functionality from [Hermes protocol](https://docs.snips.ai/reference/hermes) using [rapidfuzz](https://github.com/rhasspy/rapidfuzz).

## Requirements

* Python 3.7

## Installation

```bash
$ git clone https://github.com/rhasspy/rhasspy-fuzzywuzzy-hermes
$ cd rhasspy-fuzzywuzzy-hermes
$ ./configure
$ make
$ make install
```

## Deployment

```bash
$ make dist
```

See `dist/` directory for `.tar.gz` file.

## Running

```bash
$ bin/rhasspy-fuzzywuzzy-hermes <ARGS>
```

## Command-Line Options

```
usage: rhasspy-fuzzywuzzy-hermes [-h] [--examples EXAMPLES]
                                 [--intent-graph INTENT_GRAPH]
                                 [--casing {upper,lower,ignore}]
                                 [--replace-numbers] [--language LANGUAGE]
                                 [--confidence-threshold CONFIDENCE_THRESHOLD]
                                 [--host HOST] [--port PORT]
                                 [--username USERNAME] [--password PASSWORD]
                                 [--tls] [--tls-ca-certs TLS_CA_CERTS]
                                 [--tls-certfile TLS_CERTFILE]
                                 [--tls-keyfile TLS_KEYFILE]
                                 [--tls-cert-reqs {CERT_REQUIRED,CERT_OPTIONAL,CERT_NONE}]
                                 [--tls-version TLS_VERSION]
                                 [--tls-ciphers TLS_CIPHERS]
                                 [--site-id SITE_ID] [--debug]
                                 [--log-format LOG_FORMAT]

optional arguments:
  -h, --help            show this help message and exit
  --examples EXAMPLES   Path to examples JSON file
  --intent-graph INTENT_GRAPH
                        Path to intent graph (gzipped pickle)
  --casing {upper,lower,ignore}
                        Case transformation for input text (default: ignore)
  --replace-numbers     Replace digits with words in queries (75 -> seventy
                        five)
  --language LANGUAGE   Language/locale used for number replacement (default:
                        en)
  --confidence-threshold CONFIDENCE_THRESHOLD
                        Minimum confidence needed before intent not recognized
                        (default: 0)
  --host HOST           MQTT host (default: localhost)
  --port PORT           MQTT port (default: 1883)
  --username USERNAME   MQTT username
  --password PASSWORD   MQTT password
  --tls                 Enable MQTT TLS
  --tls-ca-certs TLS_CA_CERTS
                        MQTT TLS Certificate Authority certificate files
  --tls-certfile TLS_CERTFILE
                        MQTT TLS certificate file (PEM)
  --tls-keyfile TLS_KEYFILE
                        MQTT TLS key file (PEM)
  --tls-cert-reqs {CERT_REQUIRED,CERT_OPTIONAL,CERT_NONE}
                        MQTT TLS certificate requirements (default:
                        CERT_REQUIRED)
  --tls-version TLS_VERSION
                        MQTT TLS version (default: highest)
  --tls-ciphers TLS_CIPHERS
                        MQTT TLS ciphers to use
  --site-id SITE_ID     Hermes site id(s) to listen for (default: all)
  --debug               Print DEBUG messages to the console
  --log-format LOG_FORMAT
                        Python logger format
```
