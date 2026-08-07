def make_step(module: str, system_prompt: str, user_prompt: str, response: dict) -> dict:
    return {
        "module": module,
        "prompt": {"system_prompt": system_prompt, "user_prompt": user_prompt},
        "response": response,
    }
