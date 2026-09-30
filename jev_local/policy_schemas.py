from pydantic import BaseModel, ConfigDict, Field, model_validator

from agent_training.simulator import Scenario


class PolicyAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1, max_length=80)
    name: str = Field(min_length=1, max_length=200)
    requires: int = Field(default=0, ge=0, le=4095, strict=True)
    forbids: int = Field(default=0, ge=0, le=4095, strict=True)
    sets: int = Field(default=0, ge=0, le=4095, strict=True)
    clears: int = Field(default=0, ge=0, le=4095, strict=True)
    tokens: float = Field(default=0, ge=0, le=1000000, allow_inf_nan=False)
    latency_ms: float = Field(default=0, ge=0, le=3600000, allow_inf_nan=False)
    allowed: bool = Field(default=True, strict=True)


class PolicyScenario(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(default="custom", max_length=100)
    family: str = Field(default="custom", max_length=100)
    split: str = Field(default="custom", max_length=100)
    facts: list[str] = Field(min_length=1, max_length=12)
    state: int = Field(ge=0, le=4095, strict=True)
    goal: int = Field(ge=0, le=4095, strict=True)
    actions: list[PolicyAction] = Field(default_factory=list, max_length=10)
    context: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_world(self):
        Scenario.from_dict(self.model_dump())
        return self


class PolicyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scenario: PolicyScenario
    token_weight: float = Field(default=1, ge=0, le=1000, allow_inf_nan=False)
    latency_weight: float = Field(default=0.01, ge=0, le=1000, allow_inf_nan=False)
    max_depth: int = Field(default=5, ge=1, le=5, strict=True)

    @model_validator(mode="after")
    def some_cost_weight(self):
        if self.token_weight + self.latency_weight == 0:
            raise ValueError("At least one cost weight must be positive.")
        return self
