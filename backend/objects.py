from pydantic import BaseModel
from typing import Set, Optional

# This exists to prevent the circular dependency between main nad db
class ProcessRequest(BaseModel):
    pipe_id: str
    submit_ids: Optional[Set[int]]
    student_emails: Optional[Set[str]]
    head_n_results: Optional[int]