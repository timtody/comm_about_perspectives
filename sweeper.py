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
        if self.ranges is None:
            ranges = [
                map(lambda e: round(e, 2), list(np.linspace(0, 1, self.steps)))
                for _ in self.vars
            ]
        else:
            ranges = [
                map(
                    lambda e: round(e, 2),
                    list(np.linspace(vr[1][0], vr[1][1], self.steps)),
                )
                for vr in zip(self.vars, self.ranges)
            ]
        cartesian = itertools.product(*ranges)
        for vals in cartesian:
            yield list(zip(self.vars, vals))

    def _sweep_sample(self) -> Generator:
        zeros = np.zeros(len(self.vars))
        for i, _ in enumerate(self.vars):
            z = zeros.copy()
            z[i] = 1
            yield list(zip(self.vars, z))
        for _ in range(self.steps):
            values = np.random.uniform(0, 1, len(self.vars))
            values = map(lambda x: round(x, 2), values)
            yield list(zip(self.vars, values))


if __name__ == "__main__":
    sweeper = Sweeper(["x", "y", "z", "a"], 10, mode="sample").sweep()
    for vars in sweeper:
        print(vars)