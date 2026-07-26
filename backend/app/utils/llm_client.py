"""
LLM client wrapper
Compatible with any OpenAI-compatible API (DashScope / OpenRouter / OpenAI / Gemini proxy)
"""

import json
import re
import traceback
from typing import Optional, Dict, Any, List
from openai import OpenAI

from ..config import Config

class LLMClient:
"""LLM Client"""

````
def __init__(
    self,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None
):
    # Read values from config / environment
    self.api_key = (api_key or Config.LLM_API_KEY or "").strip()
    self.base_url = (base_url or Config.LLM_BASE_URL or "").strip()
    self.model = (model or Config.LLM_MODEL_NAME or "").strip()

    self.is_unavailable = False

    # Validate API key
    invalid_values = {
        "",
        "your_api_key_here",
        "dummy",
        "placeholder",
        "null",
        "none"
    }

    if self.api_key.lower() in invalid_values:
        self.is_unavailable = True
        self.client = None
        return

    # Create OpenAI-compatible client
    self.client = OpenAI(
        api_key=self.api_key,
        base_url=self.base_url
    )

def chat(
    self,
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    max_tokens: int = 4096,
    response_format: Optional[Dict] = None
) -> str:
    """
    Send chat request

    Returns:
        Model response text
    """

    if self.is_unavailable or self.client is None:
        raise RuntimeError(
            "LLM service unavailable: no valid API key configured"
        )

    kwargs = {
        "model": self.model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if response_format:
        kwargs["response_format"] = response_format

    try:
        print("\n========== LLM CONFIG ==========")
        print("Base URL:", self.base_url)
        print("Model:", self.model)
        print("API Key:", self.api_key[:10] + "...")
        print("================================\n")

        response = self.client.chat.completions.create(**kwargs)

        print("\n========== RAW RESPONSE ==========")
        print(response)
        print("==================================\n")

        if not response.choices:
            raise ValueError("LLM returned no choices")

        content = response.choices[0].message.content

        if content is None:
            raise ValueError("LLM returned empty content")

        # Remove <think>...</think> blocks used by some models
        content = re.sub(
            r'<think>[\s\S]*?</think>',
            '',
            content
        ).strip()

        print("========== CLEANED CONTENT ==========")
        print(content[:1000])  # print first 1000 chars only
        print("=====================================\n")

        return content

    except Exception as e:
        print("\n========== LLM ERROR ==========")
        traceback.print_exc()
        print("Error Type:", type(e).__name__)
        print("Error Message:", str(e))
        print("Base URL:", self.base_url)
        print("Model:", self.model)
        print("================================\n")
        raise

def chat_json(
    self,
    messages: List[Dict[str, str]],
    temperature: float = 0.3,
    max_tokens: int = 4096
) -> Dict[str, Any]:
    """
    Send chat request and parse JSON response
    """

    response = self.chat(
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"}
    )

    # Remove markdown code fences
    cleaned_response = response.strip()
    cleaned_response = re.sub(
        r'^```(?:json)?\s*\n?',
        '',
        cleaned_response,
        flags=re.IGNORECASE
    )
    cleaned_response = re.sub(
        r'\n?```\s*$',
        '',
        cleaned_response
    )
    cleaned_response = cleaned_response.strip()

    print("\n========== JSON TO PARSE ==========")
    print(cleaned_response[:2000])  # first 2000 chars
    print("===================================\n")

    try:
        return json.loads(cleaned_response)

    except json.JSONDecodeError as e:
        print("\n========== JSON ERROR ==========")
        print("JSON Parse Error:", str(e))
        print("Raw Response:")
        print(cleaned_response)
        print("================================\n")

        raise ValueError(
            f"LLM returned invalid JSON: {cleaned_response}"
        )
````
