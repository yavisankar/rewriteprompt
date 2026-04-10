"""
Online Pipeline — Step 6
LLM Generator — Email/Text Generation using ChatGPT (gpt-4o-mini)
"""

import json
import urllib.request
import urllib.error


SYSTEM_PROMPT = """You are an expert email writing assistant trained on professional business communication. 
You write clear, contextually appropriate emails that match the requested tone, intent, and style.
Always output only the email body (and subject line if needed). No explanations."""


class LLMGenerator:

    def generate(self, rewritten_prompt: str, api_key: str, intent_ctx: dict) -> dict:
        """
        Call ChatGPT API (gpt-4o-mini) with the rewritten prompt.
        Returns generated text.
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": rewritten_prompt},
        ]

        payload = json.dumps({
            "model":       "gpt-4o-mini",
            "messages":    messages,
            "max_tokens":  800,
            "temperature": 0.7,
        }).encode("utf-8")

        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=payload,
            headers={
                "Content-Type":  "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            body = e.read().decode()
            raise RuntimeError(f"OpenAI API error {e.code}: {body}")

        text   = data["choices"][0]["message"]["content"].strip()
        tokens = data.get("usage", {})

        return {
            "text":             text,
            "model":            data.get("model","gpt-4o-mini"),
            "prompt_tokens":    tokens.get("prompt_tokens",0),
            "completion_tokens":tokens.get("completion_tokens",0),
        }
