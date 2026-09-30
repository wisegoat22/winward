"""Bounded, reviewable text interpretation; no generated executable actions."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Fact = Literal["context_available", "requirements_clear", "reproduced", "cause_known",
               "change_applied", "focused_checks_passed", "acceptance_passed", "review_complete"]
TaskKind = Literal["bugfix", "feature", "refactor", "investigate", "review", "documentation", "tests"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)


class SituationInput(StrictModel):
    situation: str = Field(min_length=10, max_length=5000)
    goal: str = Field(min_length=3, max_length=500)


class ConfirmedFact(StrictModel):
    fact: Fact
    evidence: str = Field(min_length=2, max_length=600)


class Restriction(StrictModel):
    kind: Literal["no_edits", "no_running_tests"]
    evidence: str = Field(min_length=2, max_length=600)


class SituationDraft(StrictModel):
    domain: Literal["software", "unclear", "unsupported"]
    task_kind: TaskKind
    summary: str = Field(min_length=1, max_length=1000)
    confirmed: list[ConfirmedFact] = Field(max_length=8)
    restrictions: list[Restriction] = Field(max_length=2)
    unknowns: list[str] = Field(max_length=6)

    @model_validator(mode="after")
    def bounded_unique_items(self):
        if len({item.fact for item in self.confirmed}) != len(self.confirmed):
            raise ValueError("A fact can be confirmed only once.")
        if len({item.kind for item in self.restrictions}) != len(self.restrictions):
            raise ValueError("A restriction can appear only once.")
        if any(not value.strip() or len(value) > 500 for value in self.unknowns):
            raise ValueError("Each missing-information question must contain 1–500 characters.")
        return self

    def check_evidence(self, situation, goal):
        if any(item.evidence not in situation and item.evidence not in goal
               for item in [*self.confirmed, *self.restrictions]):
            raise ValueError("An interpreted fact or restriction lacks an exact quote from your description. Please try rephrasing it.")
        return self


class SituationDecision(SituationInput):
    draft: SituationDraft
    confirmed_facts: list[Fact] = Field(max_length=8)
    max_depth: int = Field(default=5, ge=1, le=5)

    @model_validator(mode="after")
    def valid_review(self):
        self.draft.check_evidence(self.situation, self.goal)
        if len(set(self.confirmed_facts)) != len(self.confirmed_facts):
            raise ValueError("Confirmed facts must be unique.")
        return self
