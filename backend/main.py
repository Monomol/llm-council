"""FastAPI backend for LLM Council."""

import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import uuid
from openai.types import FileObject
from fastapi.responses import Response
from openai.types.chat import ChatCompletion, ChatCompletionMessage, ChatCompletionMessageParam, ChatCompletionUserMessageParam
from openai.types.chat.chat_completion import Choice, ChoiceLogprobs

from . import storage
from .council import run_full_council

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


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "LLM Council API"}

# # TODO: no authetication is done here, possible other users
# # can maliciously access other users's messages
# @app.get("/v1/files/{conversation_id}", response_model=FileObject)
# async def retrieve_file_metadata(conversation_id: str):
#     file_object = storage.get_conversation_file_object(conversation_id)
#     return file_object if file_object is not None else ValueError("No such conversation exists.")

# # TODO: no authetication is done here, possible other users
# # can maliciously access other users's messages
# @app.get("/v1/files/{conversation_id}/content")
# async def get_file_content(conversation_id: str):
#     content = storage.get_conversation_content(conversation_id)
#     if content is None:
#         raise ValueError("No such conversation exists.")
#     return Response(
#         content=content, 
#         media_type="application/octet-stream",
#         headers={"Content-Disposition": f"attachment; filename=full_evaluation_{conversation_id}.txt"}
#     )


# TODO: should I log users here?
@app.post("/v1/chat/completions", response_model=ChatCompletion)
async def send_message(request: SendMessageRequest):
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


    conversation_id = str(uuid.uuid4())


    storage.add_user_message(conversation_id, user_prompt)

    # Run the 3-stage council process
    stage1_results, stage2_results, stage3_result, metadata = await run_full_council(
        user_prompt
    )

    # Add assistant message with all stages
    storage.add_assistant_message(
        conversation_id,
        stage1_results,
        stage2_results,
        stage3_result
    )

    stage3_result

    #     {
    #     "stage1": stage1_results,
    #     "stage2": stage2_results,
    #     "stage3": stage3_result,
    #     "metadata": metadata
    # }

    # Return the complete response with metadata
    return ChatCompletion(
        id="chat-123",
        object="chat.completion",
        created=int(time.time()),
        model="llm-council",
        choices=[
            Choice(
                index=0,
                message=
                    ChatCompletionMessage(
                        role="assistant",
                        content=[{
                            "type": "text",
                            "text": f"The final evaluation is:\n\n{stage3_result}\n\nYou can download full evaluation sitting below:"
                        },
                        {
                            "type": "file",
                            "file": {
                                "name": "full_evalutation_{conversation_id}.txt",
                                "data": storage.get_conversation_content_b64(conversation_id),
                                "mime_type": "text/plain"
                            }
                        }],
                    ),
                finish_reason="stop",
                logprobs=None
            )
        ],
        usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="::", port=8001)
