"""Keep the optional Qwen demo out of the scratch-policy startup path."""
from .engine import Engine


class ModelNotReadyError(RuntimeError):
    pass


class LazyEngine:
    def __init__(self, factory=Engine):
        self.factory = factory
        self.engine = None

    def _load(self):
        if self.engine is None:
            from huggingface_hub.errors import LocalEntryNotFoundError
            try:
                self.engine = self.factory()
            except (LocalEntryNotFoundError, FileNotFoundError) as error:
                raise ModelNotReadyError(
                    "The optional Qwen demo is not installed. Run scripts/download_model.py to install it. "
                    "Winward's trained policy does not need Qwen."
                ) from error
        return self.engine

    def decide(self, request):
        return self._load().decide(request)

    def generate(self, request):
        return self._load().generate(request)

    def tokenize(self, text, add_special_tokens=False):
        return self._load().tokenize(text, add_special_tokens)

    def score(self, request):
        return self._load().score(request)
