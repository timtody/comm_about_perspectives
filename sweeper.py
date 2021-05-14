import itertools
import numpy as np
from typing import Generator


class Sweeper:
    def __init__(
        self,
        vars: "list[str]",
        steps: int,
        ranges: "list[tuple]" = None,
        mode: str = "grid",
    ) -> None:
        if ranges is not None:
            assert len(vars) == len(
                ranges
            ), "If ranges are supplied, they need to have the same length as vars"
        assert mode == "grid" or mode == "sample", "Mode needs to be grid or sample."
        self.mode = mode
        self.vars = vars
        self.steps = steps
        self.ranges = ranges

    def sweep(self) -> Generator:
        if self.mode == "grid":
            return self._sweep_grid()
        elif self.mode == "sample":
            return self._sweep_sample()

    def _sweep_grid(self) -> Generator:
        ranges = [
            map(lambda e: round(e, 2), list(np.linspace(0, 1, self.steps)))
            for _ in self.vars
        ]
        cartesian = itertools.product(*ranges)
        for vals in cartesian:
            yield list(zip(self.vars, vals))

    def _sweep_sample(self) -> Generator:
        raise NotImplementedError


if __name__ == "__main__":
    sweeper = Sweeper(["x", "y", "z", "a"], 5).sweep()
    print(len(list(sweeper)))