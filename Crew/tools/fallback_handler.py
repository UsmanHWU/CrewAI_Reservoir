"""
Fallback Handler
Catches tool failures, empty responses, and LLM errors.
Returns graceful defaults so the crew doesn't crash.
"""


def safe_tool_run(tool_func, *args, **kwargs):
    """
    Wraps any tool call in a try-except.
    If the tool fails, returns a default fallback message.
    """
    try:
        result = tool_func(*args, **kwargs)
        if not result or result.strip() == "":
            return "[FALLBACK] Tool returned empty response. Using cached/default values."
        return result
    except TimeoutError:
        return "[FALLBACK] Tool timed out. Proceeding with estimated values."
    except ConnectionError:
        return "[FALLBACK] Network error. Check your API key or internet connection."
    except Exception as e:
        return f"[FALLBACK] Unexpected error: {str(e)}. Workflow continues with defaults."


def validate_llm_output(output: str, required_keywords: list) -> tuple[bool, str]:
    """
    Checks if the LLM output contains expected content.
    Returns (is_valid, message).
    """
    if not output or len(output.strip()) < 50:
        return False, "Output too short — likely a failed generation."
    
    missing = [kw for kw in required_keywords if kw.lower() not in output.lower()]
    if missing:
        return False, f"Output missing expected sections: {missing}"
    
    return True, "Output validated successfully."
