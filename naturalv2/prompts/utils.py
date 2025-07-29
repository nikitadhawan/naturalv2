import logging
import os
from typing import Any, Literal

import jinja2
import yaml


logger = logging.getLogger(__name__)


class PreserveUndefined(jinja2.Undefined):
    """A custom Jinja2 Undefined class that preserves the name of the undefined variable."""

    def __str__(self) -> str:
        """Return a string representation of the undefined variable."""
        return f"{{{{ {self._undefined_name} }}}}"


# The environment is created once when this module is first imported.
jinja_env = jinja2.Environment(undefined=PreserveUndefined)


def load_prompt(
    base_dir: str,
    prompt_type: str,
    return_format: Literal["messages", "prompt"] | None = None,
    **user_prompt_format_kwargs: Any,
) -> str | list[dict[str, str]]:
    """Load a prompt from a YAML file and format it with Jinja2.

    Parameters
    ----------
    base_dir : str
        The base directory where the prompt YAML files are located.
    prompt_type : str
        The type of prompt to load. This is the filename without the .yaml extension.
    return_format : str, optional, default=None
        The format to return the prompt in. Can be 'messages' for a list of dictionaries
        with 'role' and 'content' keys, 'prompt' for a single string, or None for just the
        user prompt template.
    user_prompt_format_kwargs : Any
        Additional keyword arguments to format the user prompt template with Jinja2.

    Returns
    -------
    str | list[dict[str, str]]
        If return_format is 'messages', returns a list of dictionaries with 'role' and
        'content' keys. If return_format is 'prompt', returns a single formatted string.
        If return_format is None, returns just the user prompt template as a string.

    """
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


def get_common_name_prompts(
    attribute: Literal["treatment", "outcome"],
    source: str,
    **prompt_format_kwargs: Any,
) -> list[dict[str, str]]:
    """Get common name prompts based on the attribute.

    Parameters
    ----------
    attribute : Literal["treatment", "outcome"]
        The attribute for which to get the common name prompts.
    source : str
        The source/dataset of the prompts, used for loading the correct template.
    prompt_format_kwargs : Any
        Additional keyword arguments to format the prompt with Jinja2.

    Returns
    -------
    list[dict[str, str]]
        A list of dictionaries with 'role' and 'content' keys for the common name prompts.
    """
    base_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "prompts",
        "templates",
    )

    return load_prompt(
        base_dir,
        "common_name_treatment" if attribute == "treatment" else "common_name_outcome",
        return_format="messages",
        source=source,
        **prompt_format_kwargs,
    )
