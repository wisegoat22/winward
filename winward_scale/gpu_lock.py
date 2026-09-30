"""Cross-process local lock so UI inference cannot overlap large training."""
from contextlib import contextmanager
import fcntl
from pathlib import Path


class ComputeBusy(ValueError):
    pass


@contextmanager
def local_compute():
    directory = Path(__file__).resolve().parents[1] / ".runtime"
    directory.mkdir(exist_ok=True)
    with (directory / "metal.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ComputeBusy("Local training or inference is using the Mac. Follow training at /scale and try again after it finishes.") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def run_local(method, *args):
    with local_compute():
        return method(*args)
