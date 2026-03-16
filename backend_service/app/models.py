from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    awaiting_review = "awaiting_review"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class StepState(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


class StepProgress(BaseModel):
    key: str
    label: str
    order: int
    status: StepState = StepState.pending


class Artifact(BaseModel):
    name: str
    path: str
    exists: bool = False


class ReviewableEvent(BaseModel):
    id: int
    type: str = "unknown"
    justification: str = ""
    snippet: str = ""
    confidence_score: str = ""
    date_time: str = ""
    location: str = ""
    parties: List[str] = Field(default_factory=list)
    narrative: str = ""
    type_specific_fields: Dict[str, Any] = Field(default_factory=dict)
    source_file: str = "UNKNOWN"
    selected: bool = True


class EventTypeDefinition(BaseModel):
    description: str
    specific_fields: Dict[str, str] = Field(default_factory=dict)


class EventTypeUpsertRequest(BaseModel):
    key: str
    description: str
    specific_fields: Dict[str, str] = Field(default_factory=dict)


class EventTypeListResponse(BaseModel):
    event_types: Dict[str, EventTypeDefinition]


class EventReviewSubmitRequest(BaseModel):
    selected_ids: List[int] = Field(default_factory=list)


class JobRecord(BaseModel):
    job_id: str
    created_at: str
    updated_at: str
    status: JobStatus
    current_step: str = "queued"
    step_index: int = 0
    total_steps: int = 6
    progress_percent: int = 0
    message: str = "Job created"
    cancel_requested: bool = False
    review_required: bool = False
    review_submitted: bool = False
    input_files: List[str] = Field(default_factory=list)
    artifacts: List[Artifact] = Field(default_factory=list)
    steps: List[StepProgress] = Field(default_factory=list)
    review_events: List[ReviewableEvent] = Field(default_factory=list)
    error: Optional[str] = None


class JobCreateResponse(BaseModel):
    job_id: str
    status: JobStatus


class JobListResponse(BaseModel):
    jobs: List[JobRecord]
