"""A demo experiment.

Read this script to understand how experiments work in this codebase. Usage:

```
uv run -m lib.examples.demo --help
uv run -m lib.examples.demo experiments/examples/demo --force
```
"""

from pathlib import Path
from typing import TypedDict

import lib.experiment
import lib.utils


class Config(TypedDict):
    """The experiment config.

    The config must be JSON-serializable (even though it is stored in TOML).
    """

    a: float
    b: float


def main(config: Config, exp: str | Path) -> lib.experiment.Report:
    """The experiment function.

    Args:
        config: the experiment config.
        exp: the experiment directory. It must contain the `config.json` config file.
            All experiment artifacts, and particularly the report, will also be stored
            in this directory.
    """

    # >>> Start the experiment.
    exp = Path(exp)
    # The report stores the main results of the experiments.
    report = lib.experiment.create_report(main, add_gpu_info=True)

    # >>> Do the job
    report['c'] = config['a'] + config['b']

    # >>> Finish the experiment.
    lib.experiment.finish(exp, report)
    return report


if __name__ == '__main__':
    # Configure the environment and libraries.
    lib.utils.init()
    # Run the experiment through a CLI.
    lib.experiment.run_cli(main)
