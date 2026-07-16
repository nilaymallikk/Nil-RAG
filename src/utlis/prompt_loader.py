from pathlib import Path


def load_prompt(prompt_name: str):

    prompt_path = Path("prompts") / prompt_name

    with open(prompt_path, "r") as file:

        return file.read()