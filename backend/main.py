"""FastAPI backend for LLM Council."""

from datetime import datetime
import time
from fastapi import FastAPI, Depends, HTTPException, Request, status, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uuid
import logging
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from . import storage
from .council import run_full_council, INTERACTIVE_LEARNING_SYSTEM_PROMPT
from .db import get_submissions
from .config import ALLOWED_TOKENS
from .objects import ProcessPayload

logging.basicConfig(
    filename="data/logs/log.txt",
    level=logging.INFO,
    format="%(asctime)s [%(name)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",

)

logger = logging.getLogger(__name__)

# Due to the limiter, we have to inject fastapi_request: Request
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="LLM Council API")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

security = HTTPBearer()

def validate_token(request: Request, credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials not in ALLOWED_TOKENS:
        # We use a generic error to prevent "leaking" which tokens are valid
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials


def switch_user_assistant(transcript: str):
    return "### ASSISTANT".join(map(lambda x: x.replace("### ASSISTANT", "### USER"), transcript.split("### USER")))

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "LLM Council API"}
    

@app.post("/process", status_code=status.HTTP_200_OK, dependencies=[Depends(validate_token)])
@limiter.limit("100/minute")
async def process(payload: ProcessPayload, request: Request, background_tasks: BackgroundTasks) -> dict:
    """
    Send a message and run the 3-stage council process.
    Returns the complete response with all stages.
    """

    trace_id = str(uuid.uuid4())[:8]
    start_time = time.perf_counter()
    
    # Meaningful Start Log: capture input scope without dumping huge lists
    logger.info(
        f"[{trace_id}] Started processing | pipe_id: {payload.pipe_id} | "
        f"filter_ids: {len(payload.submit_ids) if payload.submit_ids else 0} | "
        f"filter_emails: {len(payload.student_emails) if payload.student_emails else 0}"
    )

    try:
        sumbissions = get_submissions(payload)
        successfully_processed = []
        
        duration = time.perf_counter() - start_time
        logger.info(
            f"[{trace_id}] Query successful | returned: {len(sumbissions)} records | "
            f"duration: {duration:.4f}s"
        )

        for submission in sumbissions:
    
            submission_start = time.perf_counter()

            # TODO: hopefully, we should be able to remove this soon
            if payload.pipe_id == "pb160_week02_04_exam" and submission.created_at < datetime(2026, 3, 16):
                submission.transcript = switch_user_assistant(submission.transcript)

            logger.info(f"[{trace_id}] Sub-task: Starting Council for Student {submission.email} (ID: {submission.id})")
            storage.create_conversation(submission.id)
            storage.add_user_message(submission.id, submission.transcript)

            # Run the 3-stage council process
            stage1_results, stage2_results, stage3_result, metadata = await run_full_council(
                submission.transcript,
                INTERACTIVE_LEARNING_SYSTEM_PROMPT
            )

            full_evaluation = storage.add_assistant_message(
                submission.id,
                stage1_results,
                stage2_results,
                stage3_result,
                metadata
            )

            submission_duration = time.perf_counter() - submission_start
            logger.info(f"[{trace_id}] Submission {submission.id} processed successfully | duration: {submission_duration:.2f}s")

            final_result = stage3_result['response']
            successfully_processed.append(submission.id)

            # TODO: implement the email sending
            # background_tasks.add_task(upload_into_vault, full_evaluation, user_uco)
        return {"status": "OK", "processed": successfully_processed}

    except Exception as e:
        logger.error(f"[{trace_id}] Critical failure: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing failed (successfully processed submissions={successfully_processed})")



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="::", port=8001)
