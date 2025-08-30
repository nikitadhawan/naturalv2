"""Helper functions for using vLLM for offline inference."""

import warnings
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from vllm.entrypoints.llm import LLM
    from vllm.outputs import RequestOutput
    from vllm.sampling_params import SamplingParams
    from vllm.transformers_utils.tokenizer import AnyTokenizer


def get_llm_and_sampling_params(
    llm_params: dict[str, Any],
    sampling_params: dict[str, Any] | None = None,
) -> tuple["LLM", "SamplingParams"]:
    """Return an instance of `vllm.LLM` class and sampling parameters.

    Parameters
    ----------
    llm_params : dict[str, Any]
        Parameters to initialize the `vllm.LLM` class.
    sampling_params : dict[str, Any] | None, optional, default=None
        Sampling parameters to override the default ones.

    Returns
    -------
    tuple["LLM", "SamplingParams"]
        An instance of `vllm.LLM` class and sampling parameters.
    """
    import msgspec  # type: ignore # noqa: PLC0415
    from vllm.entrypoints.llm import LLM  # type: ignore # noqa: PLC0415
    from vllm.sampling_params import SamplingParams  # type: ignore # noqa: PLC0415

    llm = LLM(**llm_params)
    default_sampling_params = msgspec.structs.asdict(llm.get_default_sampling_params())

    if sampling_params is not None:
        default_sampling_params.update(sampling_params)

    return llm, SamplingParams(**default_sampling_params)


def should_add_bos(tokenizer: "AnyTokenizer") -> bool:
    """Check if the tokenizer has a beginning-of-sequence (BOS) token.

    If not, it sets the BOS token to be the same as the end-of-sequence (EOS) token
    and issues a warning.

    Parameters
    ----------
    tokenizer : AnyTokenizer
        The tokenizer to check.

    Returns
    -------
    bool
        True if the BOS token was added, False otherwise.

    Warns
    -----
    UserWarning
        If the tokenizer does not have a BOS token and it is set to the EOS token.
    """
    test_tokens = tokenizer.encode("test", add_special_tokens=True)
    if tokenizer.bos_token_id not in test_tokens:
        if tokenizer.bos_token is None:
            tokenizer.bos_token = tokenizer.eos_token
        warnings.warn(
            f"Adding to the prompt the ``bos`` token: {tokenizer.bos_token}. "
            "If this is an ``eos`` token, this tokenizer does not have a ``bos`` token.",
            stacklevel=2,
        )
        return True

    return False


def get_prompt_logprobs(
    tokenizer: "AnyTokenizer", outputs: list["RequestOutput"]
) -> list[list[float]]:
    """Extract the log probabilities of the input tokens from the outputs.

    Parameters
    ----------
    tokenizer : AnyTokenizer
        The tokenizer used to encode the prompts.
    outputs : list[RequestOutput]
        The outputs from the LLM containing the prompt log probabilities.

    Returns
    -------
    list[list[float]]
        A list of lists containing the log probabilities of the input tokens for
        each output.
    """
    logprobs = []
    for output in outputs:
        input_tokens = tokenizer.encode(output.prompt, add_special_tokens=True)
        logprobs.append(
            [i[j].logprob for i, j in zip(output.prompt_logprobs[1:], input_tokens[1:])]
        )

    return logprobs
