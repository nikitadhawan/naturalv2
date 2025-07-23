import logging
import os
from typing import Any, Literal

import jinja2
import yaml


logger = logging.getLogger(__name__)


class PreserveUndefined(jinja2.Undefined):
    """A custom Jinja2 Undefined class that preserves the name of the undefined variable."""

    def __str__(self):
        return f"{{{{ {self._undefined_name} }}}}"


# The environment is created once when this module is first imported.
jinja_env = jinja2.Environment(undefined=PreserveUndefined)


def load_prompt(
    base_dir: str,
    prompt_type: str,
    return_format: Literal["messages", "prompt"] | None = None,
    **user_prompt_format_kwargs: Any,
) -> str | list[dict[str, str]]:
    if return_format not in ["messages", "prompt", None]:
        raise ValueError("return_format must be either 'messages', 'prompt', or None.")

    filepath = os.path.join(base_dir, f"{prompt_type}.yaml")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Prompt file {filepath} not found.")

    # load yaml file
    with open(filepath, "r") as stream:
        prompt_data: dict[str, Any] = yaml.safe_load(stream)

    # 'user_prompt_template' must be in the yaml file
    if "user_prompt_template" not in prompt_data:
        raise KeyError(
            f"'user_prompt_template' key not found in {prompt_type}.yaml file."
            "Please check the file format."
        )

    # 'system_prompt_template' is optional, but if it exists, construct and return
    # a list of dictionaries with 'role' and 'content' keys
    system_prompt: str | None = prompt_data.get("system_prompt_template")
    if system_prompt:
        logger.debug(f"System prompt loaded from {prompt_type}.yaml: {system_prompt}")

        system_role_dict = {"role": "system", "content": system_prompt}

    user_prompt_template: str = prompt_data["user_prompt_template"]
    if user_prompt_format_kwargs:
        user_prompt_template = jinja_env.from_string(user_prompt_template).render(
            **user_prompt_format_kwargs
        )

    logger.debug(
        f"User prompt template loaded from {prompt_type}.yaml: {user_prompt_template}"
    )

    if "examples_intro" in prompt_data and (
        "examples" in prompt_data and len(prompt_data["examples"]) > 0
    ):
        examples_intro = prompt_data["examples_intro"]
        examples: list[dict[str, str]] = prompt_data["examples"]

        # append examples intro and examples to user_prompt_template
        user_prompt_template += f"\n\n{examples_intro}"
        for example in examples:
            user_prompt_template += f"\n\n{example['input']}\n\n{example['output']}"

    if return_format in ["messages", None]:
        user_role_dict = {"role": "user", "content": user_prompt_template}

        if system_prompt:
            return [system_role_dict, user_role_dict]
        if return_format == "messages":
            return [user_role_dict]

    # concatenate system_prompt and user_prompt_template
    # and return as a single string
    if system_prompt:
        return f"{system_prompt}\n\n{user_prompt_template}"

    logger.debug(f"Returning user prompt template as a string: {user_prompt_template}")

    return user_prompt_template
