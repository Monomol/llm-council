"""FastAPI backend for LLM Council."""

import time
import datetime
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import uuid
import httpx
from openai.types.chat import ChatCompletion, ChatCompletionMessage, ChatCompletionMessageParam
from openai.types.chat.chat_completion import Choice

from . import storage
from .council import run_full_council, INTERACTIVE_LEARNING_SYSTEM_PROMPT

app = FastAPI(title="LLM Council API")

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SendMessageRequest(BaseModel):
    messages: List[ChatCompletionMessageParam]

SUBMISSION_VAULT_URL = "https://is.muni.cz/dok/depository_in"

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "LLM Council API"}
    

# TODO: should I log users here?
@app.post("/v1/chat/completions", response_model=ChatCompletion)
async def send_message(request: SendMessageRequest, background_tasks: BackgroundTasks):
    """
    Send a message and run the 3-stage council process.
    Returns the complete response with all stages.
    """
    for msg in request.messages:
        if msg.get("role") == "user":
            user_prompt = msg["content"]
            break
    else:
        raise ValueError("No user input prompt found.")

    try:
        user_uco = user_prompt[:user_prompt.find("\n")].split("@")[0]
    except:
        # TODO: examine with Kuba, what does such response look like and how does it get handled
        raise ValueError("Could not get user's UČO.")


    conversation_id = f"{uuid.uuid4()}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}"

    storage.create_conversation(conversation_id)
    storage.add_user_message(conversation_id, user_prompt)

    # Run the 3-stage council process
    stage1_results, stage2_results, stage3_result, metadata = await run_full_council(
        user_prompt,
        INTERACTIVE_LEARNING_SYSTEM_PROMPT
    )

    full_evaluation = storage.add_assistant_message(
        conversation_id,
        stage1_results,
        stage2_results,
        stage3_result,
        metadata
    )

    final_result = stage3_result['response']

    background_tasks.add_task(upload_into_vault, full_evaluation, user_uco)

    # Return the complete response with metadata
    return ChatCompletion(
        id=conversation_id,
        object="chat.completion",
        created=int(time.time()),
        model="llm-council",
        choices=[
            Choice(
                index=0,
                message=
                    ChatCompletionMessage(
                        role="assistant",
                        content=f"The final assessment is:\n\n{final_result}\n\nThe full evaluation is being uploaded into your submission vault.",
                    ),
                finish_reason="stop",
            )
        ],
    )

async def upload_into_vault(full_evaluation: str, uco : str):
    params = {
        "vybos_vzorek_last": "",
        "vybos_vzorek": uco,
        "vybos_hledej": "Vyhledat osobu"
    }

    data = {
        "quco": uco,
        "vlsozav": "najax",
        "ajax-upload": "ajax",
        "A_POPIS_1": "processed by LLM Council"
    }

    files = {
        "FILE_1": ("full_evaluation.json", full_evaluation.encode(), "text/plain")
    }

    httpx.post(
        SUBMISSION_VAULT_URL,
        params=params,
        data=data,
        files=files
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="::", port=8001)
