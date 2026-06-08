"""
Langfuse Observability Setup
Tracks every agent action, LLM call, tool use, latency, and cost.
"""

import os
from langfuse.callback import CallbackHandler


def get_langfuse_handler():
    """
    Returns a Langfuse CallbackHandler.
    If keys are missing, prints a warning and returns None.
    CrewAI will still run — just without tracing.
    """
    pub_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    sec_key = os.getenv("LANGFUSE_SECRET_KEY")

    if not pub_key or not sec_key:
        print(
            "[WARNING] Langfuse keys not found in .env. "
            "Monitoring disabled. Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY to enable."
        )
        return None

    handler = CallbackHandler(
        public_key=pub_key,
        secret_key=sec_key,
        host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),  # or self-hosted URL
    )
    print("[INFO] Langfuse observability enabled. Visit https://cloud.langfuse.com to view traces.")
    return handler
