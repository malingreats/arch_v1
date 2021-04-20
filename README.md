## Requirements

* docker with docker-compose

## Building the containers
```sh
make build
make up
# or
make all # builds, brings containers up, runs tests
```

## Creating a local virtualenv (optional)

```sh
python3.8 -m venv .venv && source .venv/bin/activate # or however you like to create virtualenvs

pip install pytest 


pip install pytest sqlalchemy


pip install requirements.txt


pip install requirements.txt
pip install -e src/
```


## Running the tests

```sh
make test
# or, to run individual test types
make unit
make integration
make e2e
# or, if you have a local virtualenv
make up
pytest tests/unit
pytest tests/integration
pytest tests/e2e
```

## Makefile

There are more useful commands in the makefile, have a look and try them out.

# arch_v1
