#!/usr/bin/env python3
"""
author: xcizek1

A script for converting markdown transcripts into json.
The parsing is strict in ensuring the question calling structure.
Note that it is technically possible to submit a valid transcript that
fails parsing, for example if the user starts the conversation without
calling any command and the LLM then starts hallucinating questions.

Any asked question generated without calling a command is treated as malformed transcript.

Usage:
    python md2json.py <input.md> <output.json>
"""

import re
import json


def convert_transcript(text: str) -> list[dict]:
    pattern = re.compile(r"### (USER|ASSISTANT)\n")
    parts = pattern.split(text)

    turns = []
    i = 1
    while i < len(parts) - 1:
        role = parts[i].strip()
        content = parts[i + 1].strip()
        turns.append({"role": role, "content": content})
        i += 2

    # for every assistant turn that contains a "## Question" block.
    # assistant message might contain both pipeline text AND a question (info).
    # split such messages into two virtual turns, tagging the preamble
    # half as "info_preamble" so the main loop can skip it when inside a question.
    normalised = []
    for turn in turns:
        if turn["role"] == "ASSISTANT" and "## Question" in turn["content"]:
            q_match = re.search(r"(## Question.*)", turn["content"], re.DOTALL)

            if q_match is None:
                raise ValueError("No questions found in transcript.")
            
            before = turn["content"][:q_match.start()].strip()
            question_block = q_match.group(1).strip()
            if before:
                normalised.append({"role": "ASSISTANT", "content": before, "info_preamble": True})
            normalised.append({"role": "ASSISTANT", "content": question_block})
        else:
            normalised.append({"role": turn["role"], "content": turn["content"]})

    first_q_idx = None
    for idx, turn in enumerate(normalised):
        if turn["role"] == "ASSISTANT" and turn["content"].startswith("## Question"):
            first_q_idx = idx
            break

    if first_q_idx is None:
        raise ValueError("No questions found in transcript.")

    prev_user = None
    for idx in range(first_q_idx - 1, -1, -1):
        if normalised[idx]["role"] == "USER":
            prev_user = normalised[idx]["content"].strip().lower()
            break

    # the user turn immediately before the first question must be "info" or "next"
    # otherwise the model started hallucinating questions
    if prev_user not in ("info", "next"):
        raise ValueError(
            f"Malformed transcript: expected user message 'info'/'next' before first question,\n"
            f"Found: '{prev_user}'."
        )

    relevant = normalised[first_q_idx:]

    questions = []
    current_question = None
    current_qa = []
    prev_user_content = "info"

    def flush_question():
        nonlocal current_question, current_qa
        if current_question is not None:
            questions.append({
                "question": current_question,
                "conversation": current_qa,
            })
        current_question = None
        current_qa = []

    i = 0
    while i < len(relevant):
        turn = relevant[i]

        if turn["role"] == "ASSISTANT" and turn["content"].startswith("## Question"):
            question_text = None
            for line in turn["content"].split("\n"):
                if line.startswith("### "):
                    question_text = line[4:].strip()
                    break
            if question_text is None:
                question_text = turn["content"]

            # in older pipes info did not force a new question
            if current_question is not None and question_text == current_question:
                i += 1
                continue

            # check if the question was triggered because of a command and not hallucination
            # info and next bypass LLM so we can be certain only that triggers new question
            cond2 = prev_user_content.lower() in ("info", "next")
            if not cond2:
                raise ValueError(
                    f"Malformed transcript: new question block started but previous "
                    f"user message was '{prev_user_content}' (expected 'next' or 'info')."
                )

            flush_question()
            current_question = question_text
            current_qa = []

        elif turn["role"] == "USER":
            content = turn["content"].strip()
            if content.lower() == "submit":
                break
            prev_user_content = content
            if current_question is not None:
                current_qa.append({"role": "user", "content": content})

        elif turn["role"] == "ASSISTANT":
            # skip preamble half of an "info" response
            if not turn.get("info_preamble") and current_question is not None:
                current_qa.append({"role": "assistant", "content": turn["content"]})

        i += 1

    flush_question()
    for q_idx, q in enumerate(questions, start=1):
        q["question_number"] = q_idx
    return json.dumps(questions, indent=2, ensure_ascii=False)
