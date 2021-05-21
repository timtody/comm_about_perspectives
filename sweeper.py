import itertools
import numpy as np
from typing import Generator


class Sweeper:
    def __init__(
        self,
        sample_vars: "list[str]",
        grid_vars: "list[str]",
        nsamples: int,
        gridsteps: int = 0,
        ranges: "list[tuple]" = None,
        mode: str = "grid",
        warmup: bool = True,
    ) -> None:
        if ranges is not None:
            assert len(vars) == len(
                ranges
            ), "If ranges are supplied, they need to have the same length as vars"
        assert mode == "grid" or mode == "sample", "Mode needs to be grid or sample."
        self.mode = mode
        self.sample_vars = sample_vars
        self.grid_vars = grid_vars
        self.nsamples = nsamples
        self.gridsteps = gridsteps
        self.ranges = ranges
        self.warmup = warmup

    def sweep(self) -> Generator:
        if self.warmup:
            zeros = np.zeros(len(self.sample_vars))
            for i, _ in enumerate(self.sample_vars):
                one_hot = list(zeros.copy())
                one_hot[i] = 1
                round_2 = lambda x: round(x, 2)
                rounded_range = lambda: list(
                    map(round_2, list(np.linspace(0, 1, self.gridsteps)))
                )
                grid_ranges = [rounded_range() for _ in self.grid_vars]
                for vals in itertools.product(*grid_ranges):
                    yield list(
                        zip(self.sample_vars + self.grid_vars, one_hot + list(vals))
                    )

        for _ in range(self.nsamples):
            values = np.random.uniform(0, 1, len(self.sample_vars))
            values = list(map(lambda x: round(x, 2), values))
            round_2 = lambda x: round(x, 2)
            rounded_range = lambda: list(
                map(round_2, list(np.linspace(0, 1, self.gridsteps)))
            )
            grid_ranges = [rounded_range() for _ in self.grid_vars]
            for vals in itertools.product(*grid_ranges):
                yield list(zip(self.sample_vars + self.grid_vars, values + list(vals)))

    def _gridify(self, samples):
        # how can we encapsulate this?
        round_2 = lambda x: round(x, 2)
        rounded_range = lambda: map(round_2, list(np.linspace(0, 1, self.gridsteps)))
        grid_ranges = [rounded_range for _ in self.grid_vars]
        for vals in itertools.product(grid_ranges):
            yield list(zip(self.sample_vars + self.grid_vars, samples + vals))

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
    sweeper = Sweeper(
        sample_vars=["a", "b", "c", "d"],
        grid_vars=["noise"],
        nsamples=10,
        gridsteps=4,
    ).sweep()
    for vars in sweeper:
        print(vars)