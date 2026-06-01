import asyncio
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from naturalv2.models.lm import LiteLLMRouterModel


# Mock the vllm library
mock_vllm = MagicMock()
with patch.dict(
    "sys.modules",
    {
        "vllm": mock_vllm,
        "vllm.outputs": mock_vllm.outputs,
        "vllm.transformers_utils.tokenizer": mock_vllm.transformers_utils.tokenizer,
        "torch": MagicMock(),
        "msgspec": MagicMock(),
        "msgspec.structs": MagicMock(),
    },
):
    # Now that mocks are in place, we can safely import the module
    from naturalv2.models.lm import (
        LiteLLMModel,
        Model,
        VLLMModel,
    )
from naturalv2.models.types import (
    EndpointType,
    ModelInput,
)
from naturalv2.models.utils import (
    _extract_output_and_parsed,
    extract_token_usage,
    get_message_content,
)


class DummyModel(Model):
    def invoke(
        self, input_data: ModelInput, *args, endpoint: EndpointType = None, **kwargs
    ):
        endpoint = endpoint or self.endpoint
        norm, is_batch = self._validate_and_normalize_input(input_data, endpoint)
        return norm


def test_model_input_normalization():
    m = DummyModel("dummy")
    assert m.invoke("hello", endpoint="text_completion") == ["hello"]
    assert m.invoke("hi", endpoint="chat_completion") == [
        {"role": "user", "content": "hi"}
    ]


def test_model_input_validation_errors():
    m = DummyModel("dummy")
    with pytest.raises(ValueError):
        m.invoke(123, endpoint="text_completion")
    with pytest.raises(ValueError):
        m.invoke([{"role": "user"}], endpoint="chat_completion")


def test_extract_output_and_parsed():
    class DummyFormat(BaseModel):
        parsed: str

    joined_text = '{"parsed": "abcdef"}'
    out2, parsed2 = _extract_output_and_parsed([joined_text], DummyFormat, True)
    assert out2 == joined_text and parsed2.parsed == "abcdef"


def test_extract_token_usage():
    class UsageObj:
        prompt_tokens, completion_tokens, total_tokens = 1, 2, 3
        input_tokens, output_tokens = 4, 5

    tu = extract_token_usage(UsageObj())
    assert tu.prompt_tokens == 1 and tu.completion_tokens == 2
    tu2 = extract_token_usage(UsageObj(), is_responses=True)
    assert tu2.prompt_tokens == 4 and tu2.completion_tokens == 5


def test_get_message_content():
    response = {"choices": [{"message": {"content": "foo"}}]}
    assert "foo" in get_message_content(response)


def test_get_message_content_object():
    class Message:
        content = "foo"

    class Choice:
        message = Message()

    class Response:
        choices = [Choice()]

    assert "foo" in get_message_content(Response())


def test_vllmmodel_invoke_and_parse():
    mock_sampling_params = MagicMock()
    mock_sampling_params.__annotations__ = {"temperature": float, "max_tokens": int}

    mock_llm_instance = MagicMock()
    mock_llm_class = MagicMock(return_value=mock_llm_instance)
    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.return_value = "templated_prompt"
    mock_llm_instance.get_tokenizer.return_value = mock_tokenizer
    mock_llm_instance.get_default_sampling_params.return_value = MagicMock()

    mock_output = MagicMock(
        text="vllm response", token_ids=[1, 2], finish_reason="stop", logprobs=None
    )
    mock_response = MagicMock(
        request_id="req-1",
        prompt_token_ids=[1, 2, 3],
        outputs=[mock_output],
        prompt_logprobs=None,
    )
    mock_llm_instance.generate.return_value = [mock_response]

    mock_vllm = MagicMock()
    mock_vllm.LLM = mock_llm_class
    mock_vllm.SamplingParams = mock_sampling_params
    mock_vllm.transformers_utils.tokenizer.AnyTokenizer = MagicMock()

    mock_msgspec = MagicMock()
    mock_msgspec.structs.asdict.return_value = {}

    with patch.dict(
        "sys.modules",
        {
            "vllm": mock_vllm,
            "vllm.outputs": mock_vllm.outputs,
            "vllm.transformers_utils": mock_vllm.transformers_utils,
            "vllm.transformers_utils.tokenizer": mock_vllm.transformers_utils.tokenizer,
            "torch": MagicMock(),
            "msgspec": mock_msgspec,
        },
    ):
        model = VLLMModel(model_id="vllm-model")
        resp = model.invoke([{"role": "user", "content": "test"}])

    assert resp.__class__.__name__ == "ModelResponse"
    assert resp.output_text == "vllm response"


@patch("naturalv2.models.lm.VLLMModel.__init__", return_value=None)
def test_vllm_raises_notimplemented_for_ainvoke(mock_init):
    model = VLLMModel.__new__(VLLMModel)
    with pytest.raises(NotImplementedError):
        asyncio.run(model.ainvoke("test"))


def test_litellm_model_text_completion_sync():
    model = LiteLLMModel(model_id="gpt-3.5-turbo")

    # Test failure case where endpoint is wrong
    with pytest.raises(ValueError):
        model.invoke(input_data="Hello", endpoint="invalid_endpoint")

    # Test failure case where input type is wrong
    with pytest.raises(ValueError):
        model.invoke(
            input_data=[{"role": "user", "content": "Hello"}],
            endpoint="text_completion",
        )

    # Success case
    # Single string input
    resp = model.invoke(
        input_data="Why is LiteLLM amazing?",
        endpoint="text_completion",
        mock_response="LiteLLM is awesome",
    )
    assert resp.output_text == "LiteLLM is awesome"

    # List of strings input
    resp2 = model.invoke(
        input_data=["What is 2+2?", "What is the capital of France?"],
        endpoint="text_completion",
        mock_response="Mocked response",
    )

    for response in resp2:
        assert response.output_text == "Mocked response"


def test_litellm_model_chat_completion_sync():
    model = LiteLLMModel(model_id="gpt-3.5-turbo")

    # Test failure case where input type is wrong
    with pytest.raises(ValueError):
        model.invoke(
            input_data=[{"input": "Why is LiteLLM great?"}], endpoint="chat_completion"
        )

    # Success case
    # Single message as dict
    resp = model.invoke(
        input_data=[{"role": "user", "content": "Why is LiteLLM great?"}],
        endpoint="chat_completion",
        mock_response="LiteLLM is great",
    )
    assert resp.output_text == "LiteLLM is great"

    # Single string input
    resp2 = model.invoke(
        input_data="Why is the answer to life?",
        endpoint="chat_completion",
        mock_response="42",
    )
    assert resp2.output_text == "42"


def test_litellm_model_structured_output_parsing():
    class ParsedResponse(BaseModel):
        answer: str

    model = LiteLLMModel(model_id="gpt-3.5-turbo")
    resp = model.invoke(
        input_data="Parse this response",
        endpoint="chat_completion",
        response_format=ParsedResponse,
        parse_output=True,
        mock_response='{"answer": "parsed!"}',
    )
    assert resp.output_text == '{"answer": "parsed!"}'
    assert resp.output_parsed.answer == "parsed!"


def test_litelllm_router_model_invoke():
    # Test failure case where "model" is different
    with pytest.raises(ValueError):
        router = LiteLLMRouterModel(
            deployment_params={
                "azure---gpt-4": {"model": "gpt-4"},
                "openai---gpt-3.5-turbo": {"model": "gpt-3.5-turbo"},
            },
        )

    # Test success case where "model" is the same
    router = LiteLLMRouterModel(
        deployment_params={
            "azure---gpt-3.5-turbo": {"model": "openai/gpt-3.5-turbo"},
            "openai---gpt-3.5-turbo": {"model": "openai/gpt-3.5-turbo"},
        },
    )
    resp = router.invoke(
        [{"role": "user", "content": "Route this request"}],
        mock_response="This is a router-mocked response",
    )
    assert resp.output_text == "This is a router-mocked response"
