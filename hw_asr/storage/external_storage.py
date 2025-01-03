import dataclasses
import logging
from abc import abstractmethod

from hw_asr.storage.experiments_storage import RunStorage


logger = logging.getLogger(__name__)


def try_to_call_function(function, times: int = 3):
    exc = None
    for _ in range(times):
        try:
            function()
            return
        except Exception as e:
            exc = e
    logging.warning(f"Failed to use external storage: {exc}")


@dataclasses.dataclass
class CheckpointInfo:
    name: str
    creation_date: str | None = None


@dataclasses.dataclass
class RunInfo:
    checkpoints: list[CheckpointInfo]
    with_config: bool


class ExternalStorage:
    def import_config(self, run_storage: RunStorage) -> None:
        try_to_call_function(lambda: self._import_config(run_storage))

    def import_checkpoint(self, run_storage: RunStorage, checkpoint_name: str) -> None:
        try_to_call_function(lambda: self._import_checkpoint(run_storage, checkpoint_name))

    def export_config(self, run_storage: RunStorage) -> None:
        try_to_call_function(lambda: self._export_config(run_storage))

    def export_checkpoint(self, run_storage: RunStorage, checkpoint_name: str) -> None:
        try_to_call_function(lambda: self._export_checkpoint(run_storage, checkpoint_name))

    @abstractmethod
    def _import_config(self, run_storage: RunStorage) -> None:
        raise NotImplementedError()

    @abstractmethod
    def _import_checkpoint(self, run_storage: RunStorage, checkpoint_name: str) -> None:
        raise NotImplementedError()

    @abstractmethod
    def _export_config(self, run_storage: RunStorage) -> None:
        raise NotImplementedError()

    @abstractmethod
    def _export_checkpoint(self, run_storage: RunStorage, checkpoint_name: str) -> None:
        raise NotImplementedError()

    # TODO: add a method to get info about a particular run and use it in `model_loader.py`
    @abstractmethod
    def get_available_runs(self) -> dict[str, dict[str, RunInfo]]:
        raise NotImplementedError()

    @abstractmethod
    def list_content(self) -> str:
        raise NotImplementedError()
