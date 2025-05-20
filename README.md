# NATURAL-v2

This repository extends [NATURAL](https://arxiv.org/abs/2407.07018) to larger data and evaluation scales.

______________________________________________________________________

## Set-up
Prior to installing the dependencies for this project, it is recommended to install [uv](https://github.com/astral-sh/uv?tab=readme-ov-file#installation) and create a virtual environment. You may use whatever virtual environment management tool that you like, including uv, conda, and virtualenv.

```bash
git clone https://github.com/nikitadhawan/naturalv2.git
cd naturalv2
uv sync --no-cache --dev --no-build-isolation
```
**Note**: Add `--active` to the `uv sync` command if you prefer to use the active virtual environment. Otherwise, the virtual environment will be created in the `.venv` directory inside the project root.

Create a user file `conf/user/{your_name}.yaml` and add your own paths. See [nikita.yaml](https://github.com/nikitadhawan/naturalv2/tree/main/conf/user/nikita.yaml) for an example.
______________________________________________________________________

## Retrospective Study

To create a retrospective study for some `condition` (e.g. "diabetes"), with temporally split training and validation clinical trials, run:

```bash
python create_study.py condition={condition}
```

______________________________________________________________________

## Data Filtering and Curation


______________________________________________________________________

## Estimating NATURAL ATEs

To convert a curated set of Reddit data to the NATURAL-IPW ATE for the trial with NCT ID: NCT03987919, run:

```bash
python estimate_ate.py cheap_model.model_name=gpt-4.1-nano cheap_model.deployment_params.model_1=openai/natural-gpt-4.1-nano sample_model.model_name=gpt-4.1-mini sample_model.deployment_params.model_1=openai/natural-gpt-4.1-mini probs_model.model_name=gemma-3-27b-it probs_model.deployment_params.model_1=hosted_vllm/gemma-3-27b-it estimator=natural_ipw
```

Model choices can be changed based on budget. The above are the models used in the NATURAL paper. The only exception is that the LLAMA2-70B model above is the HF version, while NATURAL used Meta's official release (which hopefully doesn't matter too much).

Currently, this script is hard-coded to use the manually defined [SvT experiment](https://github.com/nikitadhawan/naturalv2/blob/main/naturalv2/evals/svt.py) corresponding to NCT ID NCT03987919. Eventually, we would like to create experiments in an automated fashion and store them as yamls, to be loaded in by this script.
