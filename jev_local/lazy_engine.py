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

    def read_situation(self, situation, goal):
        try:
            return self._load().read_situation(situation, goal)
        except ModelNotReadyError as error:
            raise ModelNotReadyError(
                "The local text helper is not installed. You can still use the preset lab at /v4. "
                "Install the optional Qwen model with scripts/download_model.py to read your own text."
            ) from error

    def tokenize(self, text, add_special_tokens=False):
        return self._load().tokenize(text, add_special_tokens)

    def score(self, request):
        return self._load().score(request)
