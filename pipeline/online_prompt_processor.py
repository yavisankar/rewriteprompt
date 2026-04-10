"""
Online Pipeline — Step 1 & 2
Prompt Preprocessing (Clean / Normalize Input)
Intent & Context Extraction (Goal, Tone, Style)
"""

import re


class PromptProcessor:

    INTENT_KW = {
        "compose":    ["write","compose","draft","create","generate","make"],
        "reply":      ["reply","respond","answer","response to"],
        "summarize":  ["summarize","summary","brief","tldr","shorten"],
        "forward":    ["forward","share","send to"],
        "request":    ["ask","request","inquire","question"],
        "follow_up":  ["follow up","following up","check in","update"],
        "complaint":  ["complaint","issue","problem","concern","unhappy"],
        "meeting":    ["meeting","schedule","calendar","invite","discuss"],
    }

    TONE_KW = {
        "formal":     ["formal","professional","official","business","sir","madam","dear"],
        "informal":   ["casual","friendly","informal","chill","hey","hi there"],
        "urgent":     ["urgent","asap","immediately","critical","emergency","today"],
        "apologetic": ["sorry","apology","apologize","regret","unfortunately"],
        "assertive":  ["must","require","demand","insist","need"],
    }

    STYLE_KW = {
        "brief":      ["short","brief","concise","quick","few words"],
        "detailed":   ["detailed","thorough","comprehensive","full","long"],
        "bullet":     ["bullet","list","points","items"],
    }

    def preprocess(self, prompt: str) -> str:
        """Clean and normalise user input."""
        text = prompt.strip()
        # Collapse extra whitespace
        text = re.sub(r"\s+", " ", text)
        # Remove invisible chars
        text = re.sub(r"[^\x20-\x7E\n]", "", text)
        return text

    def extract_intent_context(self, prompt: str) -> dict:
        """Identify intent, tone, style from the prompt text."""
        lower = prompt.lower()

        intent = next(
            (label for label, kws in self.INTENT_KW.items() if any(k in lower for k in kws)),
            "compose"
        )
        tone = next(
            (label for label, kws in self.TONE_KW.items() if any(k in lower for k in kws)),
            "professional"
        )
        style = next(
            (label for label, kws in self.STYLE_KW.items() if any(k in lower for k in kws)),
            "moderate"
        )

        # Extract rough topic (first noun phrase heuristic)
        topic_match = re.search(r"(?:about|regarding|for|on)\s+(.{5,60}?)(?:\.|,|$)", lower)
        topic = topic_match.group(1).strip() if topic_match else lower[:60]

        return {
            "intent":  intent,
            "tone":    tone,
            "style":   style,
            "topic":   topic,
            "word_count": len(prompt.split()),
        }
