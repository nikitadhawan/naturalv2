import warnings
from typing import Optional, Union

import torch
from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput
from vllm.transformers_utils.tokenizer import AnyTokenizer


class VLLM:
    def __init__(
        self,
        model: str,
        tokenizer_path: Optional[str] = None,
        trust_remote_code: bool = False,
        num_gpus: Optional[int] = None,
        dtype: str = "auto",
        seed: Optional[int] = None,
        gpu_mem_util: float = 0.9,
        enforce_eager: Optional[bool] = None,
        download_dir: Optional[str] = None,
        max_seq_len: Optional[int] = None,
        max_num_seqs: Optional[int] = None,
        add_bos: bool = False,
        # sampling params
        temperature: float = 1.0,
        top_p: float = 1.0,
        prompt_logprobs: Optional[int] = None,
        max_tokens: Optional[int] = 16,
        **kwargs,
    ):
        self.add_bos = add_bos

        if not num_gpus:
            num_gpus = torch.cuda.device_count()

        self.llm = LLM(
            model=model,
            tokenizer=tokenizer_path,
            trust_remote_code=trust_remote_code,
            tensor_parallel_size=num_gpus,
            dtype=dtype,
            seed=seed,
            gpu_memory_utilization=gpu_mem_util,
            enforce_eager=enforce_eager,
            max_model_len=max_seq_len,
            max_num_seqs=max_num_seqs,
            download_dir=download_dir,
            **kwargs,
        )

        self.tokenizer: AnyTokenizer = self.llm.get_tokenizer()
        self.check_bos()

        self._sampling_params = {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "prompt_logprobs": prompt_logprobs,
            "seed": seed,
        }

    def check_bos(self):
        # Check if tokenizer automatically adds BOS token; set add_bos to True if BOS token to be added manually
        test_tokens = self.tokenizer.encode("test", add_special_tokens=True)
        if self.tokenizer.bos_token_id not in test_tokens:
            if self.tokenizer.bos_token is None:
                self.tokenizer.bos_token = self.tokenizer.eos_token
            warnings.warn(
                f"Adding to the prompt the bos token: {self.tokenizer.bos_token}. If this is an eos token, this tokenizer does not have a bos token.",
                stacklevel=2,
            )
            self.add_bos = True

    def get_completions(
        self, prompt: Union[str, list[str]], **sampling_params
    ) -> list[RequestOutput]:
        sampling_params = SamplingParams(**{**self._sampling_params, **sampling_params})
        if isinstance(prompt, str):
            prompt = [prompt]

        if self.add_bos:
            prompt = [self.tokenizer.bos_token + p for p in prompt]

        return self.llm.generate(prompt, sampling_params)

    def get_prompt_logprobs(self, outputs: list[RequestOutput]) -> list[list[float]]:
        logprobs = []
        for output in outputs:
            input_tokens = self.tokenizer.encode(output.prompt, add_special_tokens=True)
            logprobs.append(
                [
                    i[j].logprob
                    for i, j in zip(output.prompt_logprobs[1:], input_tokens[1:])
                ]
            )

        return logprobs
