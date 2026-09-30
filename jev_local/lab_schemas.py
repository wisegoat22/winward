from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class UncertaintyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    example_id: str = Field(min_length=1, max_length=80)
    max_depth: int = Field(default=5, ge=1, le=5, strict=True)


class SandboxRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seed: int = Field(default=700001, ge=0, le=1000000000, strict=True)
    kind: str = Field(min_length=1, max_length=40, pattern=r"^[a-z_]+$")
    changed_goal: bool = Field(default=False, strict=True)
    uncertain: bool = Field(default=False, strict=True)
    policy: Literal["neural", "evidence_first", "random"] = "neural"


class V3EpisodeRequest(UncertaintyRequest):
    seed: int = Field(default=43001, ge=0, le=1000000000, strict=True)
