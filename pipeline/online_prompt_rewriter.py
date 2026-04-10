"""
Online Pipeline — Step 5
Prompt Rewriter (LLM / Rule-based Hybrid)
Builds a personalised, context-rich prompt for the LLM generator.
Also handles retry adjustments (change tone, add context, adjust style).
"""


TONE_INSTRUCTIONS = {
    "formal":     "Use a formal, professional tone with proper salutations.",
    "informal":   "Use a friendly, conversational tone.",
    "urgent":     "Communicate urgency clearly; keep the message direct and action-oriented.",
    "apologetic": "Express genuine apology; be empathetic and constructive.",
    "assertive":  "Be clear, direct, and confident.",
    "professional": "Maintain a professional and respectful tone throughout.",
}

STYLE_INSTRUCTIONS = {
    "brief":    "Keep the email concise — 3 to 5 sentences maximum.",
    "detailed": "Provide a thorough and comprehensive email.",
    "bullet":   "Use bullet points where appropriate to organise information.",
    "moderate": "Aim for a well-balanced length — not too short, not too long.",
}

INTENT_INSTRUCTIONS = {
    "compose":   "Write a complete email from scratch.",
    "reply":     "Write a professional reply to the described email.",
    "summarize": "Summarise the key points of the described content.",
    "forward":   "Draft a forwarding note that introduces the content.",
    "request":   "Craft a polite and clear request email.",
    "follow_up": "Write a professional follow-up email.",
    "complaint": "Write a constructive complaint email.",
    "meeting":   "Draft a meeting invitation or scheduling email.",
}


class PromptRewriter:

    def rewrite(self, original_prompt: str, intent_ctx: dict,
                retrieved_examples: list, use_rag: bool) -> dict:
        """
        Build a personalised system + user prompt for the LLM.
        """
        intent = intent_ctx.get("intent","compose")
        tone   = intent_ctx.get("tone","professional")
        style  = intent_ctx.get("style","moderate")
        topic  = intent_ctx.get("topic","the requested topic")

        intent_instr = INTENT_INSTRUCTIONS.get(intent, "Write an email.")
        tone_instr   = TONE_INSTRUCTIONS.get(tone, "Maintain a professional tone.")
        style_instr  = STYLE_INSTRUCTIONS.get(style, "Keep the email well-structured.")

        # Build context block from retrieved examples (RAG)
        context_block = ""
        if use_rag and retrieved_examples:
            examples_text = "\n\n---\n".join(
                [f"Example {i+1}:\n{ex[:300]}" for i, ex in enumerate(retrieved_examples[:3])]
            )
            context_block = f"""
<retrieved_context>
The following are real email examples retrieved from the knowledge base that are 
semantically similar to the request. Use them as style and tone references only — 
do not copy them verbatim:

{examples_text}
</retrieved_context>
"""

        rewritten = f"""{context_block}
<instructions>
Task: {intent_instr}
Tone: {tone_instr}
Style: {style_instr}
Topic: {topic}
</instructions>

<user_request>
{original_prompt}
</user_request>

Generate only the email content. Do not add explanations or meta-commentary.
"""

        return {"prompt": rewritten.strip(), "used_rag": use_rag}

    def adjust_for_retry(self, prev_prompt: str, scores: dict, attempt: int) -> dict:
        """
        Retry mechanism: adjust the prompt based on what scored poorly.
        """
        adjustments = []

        if scores.get("relevance_score", 1.0) < 0.5:
            adjustments.append("Focus more directly on the specific topic requested.")
        if scores.get("tone_match_score", 1.0) < 0.5:
            adjustments.append("Ensure the tone is clearly professional and appropriate.")
        if scores.get("semantic_similarity", 1.0) < 0.5:
            adjustments.append("Stay closer to the user's original intent and key phrases.")

        if attempt >= 1:
            adjustments.append("Make the email more concise and to-the-point.")

        adjustment_text = " ".join(adjustments) or "Improve the overall quality and clarity."

        new_prompt = f"""[RETRY ATTEMPT {attempt+1} — ADJUSTMENTS: {adjustment_text}]

{prev_prompt}"""

        return {"prompt": new_prompt, "used_rag": True}
