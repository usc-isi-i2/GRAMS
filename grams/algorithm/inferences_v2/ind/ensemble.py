from dataclasses import dataclass


@dataclass
class EnsembleParams:
    n_trees: int = 100


class Ensemble:
    VERSION = 100

    def __init__(self, params: EnsembleParams):
        self.params = params
