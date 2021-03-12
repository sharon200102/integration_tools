from abc import abstractmethod
from .._types import *
import pytorch_lightning as pl


class BaseProjector(pl.LightningModule):
    def __init__(self) -> None:
        super(BaseProjector, self).__init__()

    def project_the_data(self, data:Tensor) -> Tensor:
        raise NotImplementedError
