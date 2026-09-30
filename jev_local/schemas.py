from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .config import MAX_CHOICES


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Choice(StrictModel):
    name: str = Field(min_length=1, max_length=80, pattern=r"^[a-zA-Z0-9_ -]+$")
    description: str = Field(default="", max_length=500)


class DecisionRequest(StrictModel):
    state: str = Field(min_length=1, max_length=16000)
    question: str = Field(min_length=1, max_length=1000)
    choices: list[Choice] = Field(min_length=2, max_length=MAX_CHOICES)
    threshold: float = Field(default=0.8, ge=0, le=1, allow_inf_nan=False)
    margin: float = Field(default=0.2, ge=0, le=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def unique_names(self):
        names = [choice.name.casefold() for choice in self.choices]
        if len(set(names)) != len(names):
            raise ValueError("Choice names must be unique (ignoring case).")
        return self


class GenerationRequest(DecisionRequest):
    max_tokens: int = Field(default=80, ge=1, le=256)


class TokenizeRequest(StrictModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)
    prompt: str = Field(min_length=1, max_length=16000)
    model: str | None = None
    add_special_tokens: bool = False


class ScoreRequest(StrictModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)
    query: str = Field(min_length=1, max_length=16000)
    items: list[str] = Field(default_factory=lambda: [""], min_length=1, max_length=1)
    label_token_ids: list[int] = Field(min_length=2, max_length=MAX_CHOICES)
    apply_softmax: bool = True
    model: str | None = None

    @field_validator("query")
    @classmethod
    def nonblank_query(cls, value):
        if not value.strip():
            raise ValueError("Query must contain non-whitespace text.")
        return value

    @field_validator("items")
    @classmethod
    def only_empty_item(cls, value):
        if value != [""]:
            raise ValueError('This endpoint supports only items=[""].')
        return value

    @field_validator("label_token_ids")
    @classmethod
    def unique_nonnegative_ids(cls, value):
        if len(set(value)) != len(value) or any(t < 0 for t in value):
            raise ValueError("Token IDs must be unique and nonnegative.")
        return value
