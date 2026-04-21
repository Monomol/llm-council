"""3-stage LLM Council orchestration."""

from typing import List, Dict, Any, Tuple
from .openrouter import query_models_parallel, query_model
from .config import COUNCIL_MODELS, CHAIRMAN_MODEL

INTERACTIVE_LEARNING_SYSTEM_PROMPT="""\
**Role:** You are an Expert Pedagogical Evaluator. Your task is to analyze an examination transcript between a **"USER" (the student)** and an **"ASSISTANT" (the Examiner)** to determine the student's mastery of the subject matter.

**Core Objective:**
Provide a qualitative and quantitative assessment of the student's performance. You must act as an independent fact-checker, verifying the accuracy of the student's claims against objective reality. You will grade individual responses based on a specific rubric and provide grounded evidence for every judgment made in the report. All evaluations must be written in an impersonal, third-person perspective (e.g., "the student demonstrated" rather than "you demonstrated").

### 1. Grading Rubric
Evaluate each individual response from the student using the following scale. Your justification must explicitly reference these criteria:

| Grade | Criteria / Persona |
| :--- | :--- |
| **A** | **Excellent:** Deep, comprehensive mastery. Thorough, precise, and insightful answers. The student uses correct technical terminology and shows connections between concepts. |
| **B** | **Very Good:** Solid understanding. Mostly accurate and nearly complete answers with only minor gaps. The student generally uses correct terminology with occasional small imprecisions. |
| **C** | **Average:** Moderate understanding. Partially correct answers covering the main idea but missing details. The student sometimes uses imprecise language or leaves out nuances. |
| **D** | **Below-Average:** Basic, surface-level understanding. Vague grasp of the topic; the student misses most specifics and uses vague or occasionally incorrect terminology. |
| **E** | **Poor:** Very limited and frequently incorrect understanding. Answers are mostly wrong or confused; the student shows only isolated fragments of knowledge. |
| **F** | **Failing:** Essentially no understanding. Completely wrong, confused, or off-topic answers. The student shows no meaningful knowledge of the material. |

### 2. Analysis Protocol (Internal Monologue)
Before generating the final report:
1.  **Fact-Check:** Verify the technical accuracy of every student statement. Do not trust the "ASSISTANT's" validation; they may be incorrect or overly lenient.
2.  **Apply Rubric:** Match each student response to the A-F persona above.
3.  **Synthesize Question Blocks:** For each primary question, look at the grades of the initial response and all subsequent follow-ups to determine a single, weighted "Overall Question Grade."
4.  **Grounding:** Identify the specific sentence or concept in the transcript that justifies the grade and the high-level summary.

### 3. Required Output Structure

#### I. Executive Summary
A high-level overview of the examination's flow and the student's overall performance style.

#### II. Question-by-Question Evaluation

**Question #{{Number}}:** {{Verbatim_Primary_Question_Text}}
**Your Response:** “{{User_Answer}}”
**Grade:** {{Letter_Grade}} ({{Descriptor}})
**Evidence & Rationale:** {{Explanation_of_grading_logic_referencing_specifics_of_the_answer}}

**Follow-up #{{Question_Number}}.{{Follow_up_Number}}:** {{Verbatim_Follow_up_Question_Text}}
**Your Response:** “{{User_Answer}}”
**Grade:** {{Letter_Grade}} ({{Descriptor}})
**Evidence & Rationale:** {{Explanation_of_grading_logic}}

*(Repeat Follow-up block as needed)*

**Overall Question Grade:** {{Aggregate_Letter_Grade}} ({{Descriptor}})
**Overall Question Rationale:** {{A synthesis of how the primary response and follow-up responses together demonstrate the student's level of mastery for this specific question block.}}

---

#### III. Topic Breakdown
* **The Examined Topics:** A bulleted list of all distinct subjects or sub-topics covered.
* **Demonstrated Mastery:** List topics where the student showed high proficiency (Grade A/B level). To support these claims, provide evidence by referencing specific parts of the transcript and explaining how the student's responses demonstrated mastery. Always mention the corresponding question number when referring to these instances.
* **Knowledge Gaps:** List topics where the student failed or struggled (Grade D/E/F level). Provide evidence for each gap by referencing specific errors, misconceptions, or vague terminology used in the transcript. Always mention the corresponding question number when referring to these instances.

#### IV. Targeted Study Recommendations
Suggest specific areas or concepts the student should study next based strictly on the identified gaps.

### 4. Critical Constraints
* **Impersonal Tone:** Use indirect language. Refer to the examinee as "the student," "the examinee," or "the user." Avoid using "you" or "your" in the Evidence & Rationale or Summary sections.
* **Independent Judgment:** If the student gives a wrong answer but the ASSISTANT says "Correct!", you **must** still grade it as an error (E or F).
* **Evidence-Based:** Every grade and every summary point must be supported by a "why" based on the provided rubric and transcript text.
* **Being Verbatim:** Be verbatim in your report when you state the content of (follow-up) questions or responses.
* **No Total Exam Grade:** Provide an overall grade for each primary question block (including its follow-ups), but do not provide a single final numerical or letter grade for the entire examination.
* **English-Only Output:** Provide your assessment in English only.
"""

EXAM_SYSTEM_PROMPT="""\
**Role:** You are an Expert Pedagogical Evaluator. Your task is to analyze an examination transcript between a **user** (the student) and an **assistant** (the Examiner) to determine the student's mastery of the subject matter.

**Core Objective:**
Provide a qualitative and quantitative assessment of the student's performance. You must act as an independent fact-checker, verifying the accuracy of the student's claims against objective reality. You will grade individual responses based on a specific rubric and provide grounded evidence for every judgment made in the report. All evaluations must be written in an impersonal, third-person perspective (e.g., "the student demonstrated" rather than "you demonstrated").

### 1. Grading Rubric
Evaluate each individual response from the student using the following scale. Your justification must explicitly reference these criteria:

| Grade | Criteria / Persona |
| :--- | :--- |
| **A** | **Excellent:** Deep, comprehensive mastery. Thorough, precise, and insightful answers. The student uses correct technical terminology and shows connections between concepts. |
| **B** | **Very Good:** Solid understanding. Mostly accurate and nearly complete answers with only minor gaps. The student generally uses correct terminology with occasional small imprecisions. |
| **C** | **Average:** Moderate understanding. Partially correct answers covering the main idea but missing details. The student sometimes uses imprecise language or leaves out nuances. |
| **D** | **Below-Average:** Basic, surface-level understanding. Vague grasp of the topic; the student misses most specifics and uses vague or occasionally incorrect terminology. |
| **E** | **Poor:** Very limited and frequently incorrect understanding. Answers are mostly wrong or confused; the student shows only isolated fragments of knowledge. |
| **F** | **Failing:** Essentially no understanding. Completely wrong, confused, or off-topic answers. The student shows no meaningful knowledge of the material. |

### 2. Analysis Protocol (Internal Monologue)
Before generating the final report:
1.  **Fact-Check:** Verify the technical accuracy of every student statement. Do not trust the "assistant's" validation; they may be incorrect or overly lenient.
2.  **Apply Rubric:** Match each student response to the A-F persona above.
3.  **Synthesize Question Blocks:** For each primary question, look at the grades of the initial response and all subsequent follow-ups to determine a single, weighted "Overall Question Grade."
4.  **Grounding:** Identify the specific sentence or concept in the transcript that justifies the grade and the high-level summary.

### 3. Required Output Structure
The final evaluation must be provided strictly in JSON format according to the following structure:

```json
{
  "executive_summary": "A high-level overview of the examination's flow and the student's overall performance style.",
  "question_by_question_evaluation": [
    {
      "question_number": "{{Number}}",
      "primary_question_text": "{{Verbatim_Primary_Question_Text}}",
      "student_response": "{{User_Answer}}",
      "grade": "{{Letter_Grade}} ({{Descriptor}})",
      "evidence_and_rationale": "{{Explanation_of_grading_logic_referencing_specifics_of_the_answer}}",
      "follow_ups": [
        {
          "follow_up_number": "{{Question_Number}}.{{Follow_up_Number}}",
          "follow_up_question_text": "{{Verbatim_Follow_up_Question_Text}}",
          "student_response": "{{User_Answer}}",
          "grade": "{{Letter_Grade}} ({{Descriptor}})",
          "evidence_and_rationale": "{{Explanation_of_grading_logic}}"
        }
      ],
      "overall_question_grade": "{{Aggregate_Letter_Grade}} ({{Descriptor}})",
      "overall_question_rationale": "A synthesis of how the primary response and follow-up responses together demonstrate the student's level of mastery for this specific question block."
    }
  ],
  "topic_breakdown": {
    "examined_topics": [
      "A list of all distinct subjects or sub-topics covered."
    ],
    "demonstrated_mastery": [
      "Topics where the student showed high proficiency (Grade A/B level). Provide evidence by referencing specific parts of the transcript and explaining how the student's responses demonstrated mastery, mentioning the corresponding question number."
    ],
    "knowledge_gaps": [
      "Topics where the student failed or struggled (Grade D/E/F level). Provide evidence for each gap by referencing specific errors, misconceptions, or vague terminology used in the transcript, mentioning the corresponding question number."
    ]
  },
  "targeted_study_recommendations": [
    "Suggest specific areas or concepts the student should study next based strictly on the identified gaps."
  ]
}
```

### 4. Critical Constraints
* **Impersonal Tone:** Use indirect language. Refer to the examinee as "the student," "the examinee," or "the user." Avoid using "you" or "your" in the Evidence & Rationale or Summary sections.
* **Independent Judgment:** If the student gives a wrong answer but the assistant says "Correct!", you **must** still grade it as an error (E or F).
* **Evidence-Based:** Every grade and every summary point must be supported by a "why" based on the provided rubric and transcript text.
* **Being Verbatim:** Be verbatim in your report when you state the content of (follow-up) questions or responses.
* **No Total Exam Grade:** Provide an overall grade for each primary question block (including its follow-ups), but do not provide a single final numerical or letter grade for the entire examination.
* **English-Only Output:** Provide your assessment in English only.
"""

async def stage1_collect_responses(user_query: str, system_prompt: str) -> List[Dict[str, Any]]:
    """
    Stage 1: Collect individual responses from all council models.

    Args:
        user_query: The user's transcript

    Returns:
        List of dicts with 'model' and 'response' keys
    """
    messages = [{"role" : "system", "content" : system_prompt}, {"role": "user", "content": user_query}]

    # Query all models in parallel
    responses = await query_models_parallel(COUNCIL_MODELS, messages)
    if responses is None:
        return None

    # Format results
    stage1_results = []
    for model, response in responses.items():
        if response is not None:  # Only include successful responses
            stage1_results.append({
                "model": model,
                "response": response.get('content', '')
            })

    return stage1_results


async def stage2_collect_rankings(
    user_query: str,
    system_prompt: str,
    stage1_results: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Stage 2: Each model ranks the anonymized responses.

    Args:
        user_query: The original user query
        stage1_results: Results from Stage 1

    Returns:
        Tuple of (rankings list, label_to_model mapping)
    """
    # Create anonymized labels for responses (Response A, Response B, etc.)
    labels = [chr(65 + i) for i in range(len(stage1_results))]  # A, B, C, ...

    # Create mapping from label to model name
    label_to_model = {
        f"Response {label}": result['model']
        for label, result in zip(labels, stage1_results)
    }

    # Build the ranking prompt
    responses_text = "\n\n".join([
        f"Response {label}:\n{result['response']}"
        for label, result in zip(labels, stage1_results)
    ])

    ranking_prompt = f"""You are evaluating different solutions to the following task:

{system_prompt}

Provided transcript:

=== START ===
{user_query}
=== END ===

Here are the responses from different models (anonymized):

{responses_text}

Your task:
1. First, evaluate each response individually. For each response, explain what it does well and what it does poorly.
2. Then, at the very end of your response, provide a final ranking.

IMPORTANT: Your final ranking MUST be formatted EXACTLY as follows:
- Start with the line "FINAL RANKING:" (all caps, with colon)
- Then list the responses from best to worst as a numbered list
- Each line should be: number, period, space, then ONLY the response label (e.g., "1. Response A")
- Do not add any other text or explanations in the ranking section

Example of the correct format for your ENTIRE response:

Response A provides good detail on X but misses Y...
Response B is accurate but lacks depth on Z...
Response C offers the most comprehensive answer...

FINAL RANKING:
1. Response C
2. Response A
3. Response B

Now provide your evaluation and ranking:"""

    messages = [{"role": "user", "content": ranking_prompt}]

    # Get rankings from all council models in parallel
    responses = await query_models_parallel(COUNCIL_MODELS, messages)
    if responses is None:
        return None

    # Format results
    stage2_results = []
    for model, response in responses.items():
        if response is not None:
            full_text = response.get('content', '')
            parsed = parse_ranking_from_text(full_text)
            stage2_results.append({
                "model": model,
                "ranking": full_text,
                "parsed_ranking": parsed
            })

    return stage2_results, label_to_model


async def stage3_synthesize_final(
    user_query: str,
    system_prompt: str,
    stage1_results: List[Dict[str, Any]],
    stage2_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Stage 3: Chairman synthesizes final response.

    Args:
        user_query: The original user query
        stage1_results: Individual model responses from Stage 1
        stage2_results: Rankings from Stage 2

    Returns:
        Dict with 'model' and 'response' keys
    """
    # Build comprehensive context for chairman
    stage1_text = "\n\n".join([
        f"Model: {result['model']}\nResponse: {result['response']}"
        for result in stage1_results
    ])

    stage2_text = "\n\n".join([
        f"Model: {result['model']}\nRanking: {result['ranking']}"
        for result in stage2_results
    ])

    chairman_prompt = f"""You are the Chairman of an LLM Council. Multiple AI models have provided solutions to a user's task, and then ranked each other's responses.

Original Task:

{system_prompt}

Provided Trascript:

=== START ===
{user_query}
=== END ===

STAGE 1 - Individual Responses:
{stage1_text}

STAGE 2 - Peer Rankings:
{stage2_text}

Your task as Chairman is to synthesize all of this information into a single, comprehensive, accurate solution to the user's original task. Consider:
- The individual responses and their insights
- The peer rankings and what they reveal about response quality
- Any patterns of agreement or disagreement

In your response, do not comment on the ranking process (i. e., do not mention which models participated in the assessment, were the best, etc.).

Also, instead of speaking about some "USER" be direct as you are speaking to a user.

Provide a clear, well-reasoned final answer that represents the council's collective wisdom:"""

    messages = [{"role": "user", "content": chairman_prompt}]

    # Query the chairman model
    response = await query_model(CHAIRMAN_MODEL, messages)

    if response is None:
        # Fallback if chairman fails
        return None

    return {
        "model": CHAIRMAN_MODEL,
        "response": response.get('content', '')
    }


def parse_ranking_from_text(ranking_text: str) -> List[str]:
    """
    Parse the FINAL RANKING section from the model's response.

    Args:
        ranking_text: The full text response from the model

    Returns:
        List of response labels in ranked order
    """
    import re

    # Look for "FINAL RANKING:" section
    if "FINAL RANKING:" in ranking_text:
        # Extract everything after "FINAL RANKING:"
        parts = ranking_text.split("FINAL RANKING:")
        if len(parts) >= 2:
            ranking_section = parts[1]
            # Try to extract numbered list format (e.g., "1. Response A")
            # This pattern looks for: number, period, optional space, "Response X"
            numbered_matches = re.findall(r'\d+\.\s*Response [A-Z]', ranking_section)
            if numbered_matches:
                # Extract just the "Response X" part
                return [re.search(r'Response [A-Z]', m).group() for m in numbered_matches]

            # Fallback: Extract all "Response X" patterns in order
            matches = re.findall(r'Response [A-Z]', ranking_section)
            return matches

    # Fallback: try to find any "Response X" patterns in order
    matches = re.findall(r'Response [A-Z]', ranking_text)
    return matches


def calculate_aggregate_rankings(
    stage2_results: List[Dict[str, Any]],
    label_to_model: Dict[str, str]
) -> List[Dict[str, Any]]:
    """
    Calculate aggregate rankings across all models.

    Args:
        stage2_results: Rankings from each model
        label_to_model: Mapping from anonymous labels to model names

    Returns:
        List of dicts with model name and average rank, sorted best to worst
    """
    from collections import defaultdict

    # Track positions for each model
    model_positions = defaultdict(list)

    for ranking in stage2_results:
        ranking_text = ranking['ranking']

        # Parse the ranking from the structured format
        parsed_ranking = parse_ranking_from_text(ranking_text)

        for position, label in enumerate(parsed_ranking, start=1):
            if label in label_to_model:
                model_name = label_to_model[label]
                model_positions[model_name].append(position)

    # Calculate average position for each model
    aggregate = []
    for model, positions in model_positions.items():
        if positions:
            avg_rank = sum(positions) / len(positions)
            aggregate.append({
                "model": model,
                "average_rank": round(avg_rank, 2),
                "rankings_count": len(positions)
            })

    # Sort by average rank (lower is better)
    aggregate.sort(key=lambda x: x['average_rank'])

    return aggregate


async def generate_conversation_title(user_query: str) -> str:
    """
    Generate a short title for a conversation based on the first user message.

    Args:
        user_query: The first user message

    Returns:
        A short title (3-5 words)
    """
    title_prompt = f"""Generate a very short title (3-5 words maximum) that summarizes the following task.
The title should be concise and descriptive. Do not use quotes or punctuation in the title.

Question: {user_query}

Title:"""

    messages = [{"role": "user", "content": title_prompt}]

    # Use gemini-2.5-flash for title generation (fast and cheap)
    response = await query_model("google/gemini-2.5-flash", messages, timeout=30.0)

    if response is None:
        # Fallback to a generic title
        return "New Conversation"

    title = response.get('content', 'New Conversation').strip()

    # Clean up the title - remove quotes, limit length
    title = title.strip('"\'')

    # Truncate if too long
    if len(title) > 50:
        title = title[:47] + "..."

    return title


async def run_full_council(user_query: str, system_prompt: str) -> Tuple[List, List, Dict, Dict]:
    """
    Run the complete 3-stage council process.

    Args:
        user_query: The user's transcript of an examination

    Returns:
        Tuple of (stage1_results, stage2_results, stage3_result, metadata)
    """
    # Stage 1: Collect individual responses
    stage1_results = await stage1_collect_responses(user_query, system_prompt)

    # If no models responded successfully, return error
    if stage1_results is None:
        return [], [], {
            "model": "error",
            "response": "Some models failed to respond (stage1). For further info check the logs."
        }, {}

    # Stage 2: Collect rankings
    stage2_results = await stage2_collect_rankings(user_query, system_prompt, stage1_results)
    
    if stage2_results is None:
        return [], [], {
            "model": "error",
            "response": "Some models failed to respond (stage2). For further info check the logs."
        }, {}
    
    stage2_results, label_to_model = stage2_results
    

    # Calculate aggregate rankings
    aggregate_rankings = calculate_aggregate_rankings(stage2_results, label_to_model)

    # Stage 3: Synthesize final answer
    stage3_result = await stage3_synthesize_final(
        user_query,
        system_prompt,
        stage1_results,
        stage2_results
    )

    if stage3_result is None:
        return [], [], {
            "model": "error",
            "response": "Unable to generate final synthesis (stage3). For further info check the logs."
        }, {}

    # Prepare metadata
    metadata = {
        "label_to_model": label_to_model,
        "aggregate_rankings": aggregate_rankings
    }

    return stage1_results, stage2_results, stage3_result, metadata
