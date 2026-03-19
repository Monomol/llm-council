from pydantic import BaseModel
from typing import Set, Optional

# This exists to prevent the circular dependency between main nad db
class ProcessPayload(BaseModel):
    pipe_id: str
    submit_ids: Optional[Set[int]] = None
    student_emails: Optional[Set[str]] = None
    random_sample: bool = False
    head_n_results: Optional[int] = None