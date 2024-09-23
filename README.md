
This is the repository for the Value Aggregation project. It is chiefly maintained by Jared Moore (jlcmoore@stanford.edu).

## Repository Structure

- *`env-aggregation`*
    - Run `make init` to set up.
- *`external_data`*
    - Run `make data` to set up    
- `experiments`
    - `intuition`: Files and data to run experiments on humans (mturk) and llms. See [the README](intuition/README.md).
    - `prevalence`: Testing for the prevalence of different models in natural data sets.
        - `nlp_positionality.ipynb`: An example of processing the NLPositionality data and comparing formal models.
        - `moral_machines.ipynb`: An example of processing the Moral Machines data and comparing formal models.
        - `taylor_data.ipynb`: An example of processing the processed Common Sense Normbank data from Taylor's Value Kaleidoscope project.
- `src`: The files for the `value_aggregation` python package
- `demos.ipynb`: Various examples of moral uncertainty between different models.
- `Makefile`
    - `make test`: runs the test suite
    - `make init`: initializes the virtual environment and installs dependencies
    - `make data`: downloads external data into `./external_data`
- `pyproject.toml`: For setting up the model as a package.
- `README.md`
- `requirements.txt`: Python dependencies
- `setup.py`: For setting up the module as a package.

### `src` (the `value_aggregation` package)

- `value_aggregation`: The implementation of the formal models
    - `__init__.py`
    - `Game.py`: Lays out the various formal models and their parameters.
    - `GameState.py`: Constructors for various kinds of game states, used in different game set ups. For example: MEC, NBS, MFT, MFO, and the (discarded) expecti-max and equilibria selection.
    - `utils.py`: Utility functions for the package.
- `tests`: Various tests of the functionality in `parliament`
    - `__init__.py`
    - `context.py`
    - `TestGameNodes.py`
    - `TestGameState.py`
    - `TestOtherTheories.py`
    - `TestRunParliament.py`
    - `TestUtils.py`

## Running `jupyter` notebooks

[Install `jupyter`](https://jupyter.org/install).

Run `source env-parliament/bin/activate; jupyter notebook` and open any `.ipynb` file.
