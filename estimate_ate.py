import asyncio
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Literal, Optional

import hydra
import nest_asyncio
import numpy as np
import pandas as pd
from aiolimiter import AsyncLimiter
from dotenv import load_dotenv
from hydra.utils import instantiate
from omegaconf import DictConfig
from pydantic import BaseModel
from scipy.special import softmax
from tqdm import tqdm

from naturalv2.evals.svt import SvT
from naturalv2.models.lm import LM
from naturalv2.utils import (
    ImputationsResponse,
    KnownsResponse,
    TYFilterResponse,
    enum_to_dcts,
    enumerate_strings,
    get_sample_text,
    qa_interleaved_enum,
)


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

load_dotenv(".env")


async def extract_covariates(
    input_df: pd.DataFrame,
    experiment: SvT,
    model_cfg: DictConfig,
    save_path: str,
    extract_type: Literal["ty_filter", "knowns", "imputations"],
    response_format: Optional[type[BaseModel]] = None,
    api_calls_per_second: int = 8,
    save_freq: int = 100,
):
    if extract_type == "ty_filter":
        save_filename = f"{model_cfg.model.replace('/', '-')}_ty_samples.csv"
    elif extract_type == "knowns":
        save_filename = f"{model_cfg.model.replace('/', '-')}_samples_knowns.csv"
    elif extract_type == "imputations":
        save_filename = f"{model_cfg.model.replace('/', '-')}_samples_imputed.csv"
    else:
        raise ValueError(
            f"Invalid extract_type. Expected one of ['ty_filter', 'knowns', 'imputations'], "
            f"but got {extract_type}."
        )

    save_path = os.path.join(save_path, f"{experiment.nct_id}", save_filename)
    if os.path.exists(save_path):
        return pd.read_csv(save_path, index_col=0)

    system_msg = {"role": "system", "content": experiment.get_prompt(extract_type)}
    human_template = "\n## Input \n>{report}"
    model = LM(**model_cfg)
    rate_limiter = AsyncLimiter(api_calls_per_second, 1)
    concurrency_limiter = asyncio.Semaphore(api_calls_per_second)

    progress_bar = tqdm(
        total=len(input_df), desc=f"Extracting covariates ({extract_type})"
    )

    out_dicts = []
    out_dicts_lock = asyncio.Lock()

    active_save_tasks = set()

    # Function to clean up completed save tasks
    def cleanup_save_task(task):
        active_save_tasks.discard(task)
        try:
            rows_saved = task.result()
            progress_bar.write(f"Saved {rows_saved} new rows to {save_path}")
        except Exception as e:
            progress_bar.write(f"Save task failed: {e}")

    tasks = [
        asyncio.create_task(
            _extract_covariates_from_report(
                report=report,
                system_msg=system_msg,
                human_template=human_template,
                model=model,
                input_df=input_df,
                experiment=experiment,
                extract_type=extract_type,
                concurrency_limiter=concurrency_limiter,
                rate_limiter=rate_limiter,
                response_format=response_format,
                out_dicts=out_dicts,
                out_dicts_lock=out_dicts_lock,
                save_frequency=save_freq,
                save_path=save_path,
                active_save_tasks=active_save_tasks,
                cleanup_save_task=cleanup_save_task,
                progress_bar=progress_bar,
            )
        )
        for report in input_df["report"]
    ]

    await asyncio.gather(*tasks)

    if active_save_tasks:  # wait for any remaining save tasks to complete
        progress_bar.write(
            f"Waiting for {len(active_save_tasks)} remaining save tasks to complete..."
        )
        await asyncio.gather(*active_save_tasks)

    async with out_dicts_lock:  # save any remaining results
        if out_dicts:
            final_save_task = asyncio.create_task(
                _save_results_to_csv(out_dicts, save_path)
            )
            await final_save_task
            print(f"Final results saved to {save_path}")

    progress_bar.close()

    return pd.read_csv(save_path, index_col=0)


async def _extract_covariates_from_report(
    report: str,
    system_msg: dict,
    human_template: str,
    model: LM,
    input_df: pd.DataFrame,
    experiment: SvT,
    extract_type: str,
    concurrency_limiter: asyncio.Semaphore,
    rate_limiter: AsyncLimiter,
    response_format: Optional[type[BaseModel]],
    out_dicts: list[dict],
    out_dicts_lock: asyncio.Lock,
    save_frequency: int,
    save_path: str,
    active_save_tasks: set[asyncio.Task],
    cleanup_save_task: callable,
    progress_bar: tqdm,
) -> None:
    messages = [
        system_msg,
        {
            "role": "user",
            "content": human_template.format(report=report),
        },
    ]

    try:
        async with concurrency_limiter, rate_limiter:
            response = await model.apredict(
                messages=messages, response_format=response_format
            )

        parsed_response: list[dict] = [
            response_format.model_validate_json(text).model_dump() for text in response
        ]

        save_needed = False

        async with out_dicts_lock:  # thread-safe update of out_dicts
            out_dicts.extend(
                {**response, "report": report} for response in parsed_response
            )

            # Check if we should save based on the current count
            if len(out_dicts) % save_frequency == 0:
                save_needed = True

        if save_needed:  # outside the lock to avoid blocking other tasks
            save_task = asyncio.create_task(
                _save_results_to_csv(
                    out_dicts.copy(), save_path, experiment, extract_type, input_df
                )
            )

            active_save_tasks.add(save_task)
            save_task.add_done_callback(cleanup_save_task)
    except Exception as e:
        progress_bar.write(f"Error processing report: {report[:30]}...: {str(e)}")
    finally:
        progress_bar.update(n=1)


async def _save_results_to_csv(
    data: list[dict],
    save_path: str,
    experiment: SvT,
    extract_type: str,
    input_df: pd.DataFrame,
) -> int:
    loop = asyncio.get_running_loop()  # for running in ThreadPoolExecutor

    def _process_and_save():
        new_df = pd.DataFrame.from_dict(data)

        if (
            extract_type == "imputations"
        ):  # TODO later: Remove to use only new extractions - shouldn't change results much.
            input_df.update(new_df, overwrite=False)
            new_df = input_df.copy()

        if extract_type != "ty_filter":
            new_df = experiment.discretize(new_df, hard_filter=False, inf=False)

        if os.path.exists(save_path):
            try:
                existing_df = pd.read_csv(save_path)

                mask = ~new_df["report"].isin(existing_df["report"])
                unique_new_rows = new_df[mask]

                # append only unique new rows
                final_df = pd.concat([existing_df, unique_new_rows], ignore_index=True)

                final_df.to_csv(save_path)
                return len(unique_new_rows)
            except Exception as e:
                logger.error(f"Error reading existing file: {e}. Saving new file.")
                new_df.to_csv(save_path)
                return len(new_df)

        new_df.to_csv(save_path)
        return len(new_df)

    with ThreadPoolExecutor() as pool:  # to avoid blocking the event loop
        try:
            return await loop.run_in_executor(pool, _process_and_save)
        except Exception as e:
            logger.error(f"Error saving to data: {e}")
            return 0


def extract_conditionals(
    input_df: pd.DataFrame,
    experiment: SvT,
    model_cfg: DictConfig,
    save_path: str,
    extract_type: Literal["ty_given_x", "y_given_tx", "inclusion", None],
    length_norm: bool = False,
    batch_size: int = 1,
):
    # TODO for NOI: specify different possible conditionals to extract: (X \in I | X), (T,Y|X), (Y|T,X)
    if extract_type == "ty_given_x":
        save_path = os.path.join(
            save_path,
            f"{experiment.nct_id}",
            f"{model_cfg.model.replace('/', '-')}_ty_given_x.csv",
        )
        to_enum = ["treatment"] + experiment.outcome_names
    elif extract_type == "y_given_tx":
        save_path = os.path.join(
            save_path,
            f"{experiment.nct_id}",
            f"{model_cfg.model.replace('/', '-')}_y_given_tx.csv",
        )
        to_enum = experiment.outcome_names
    elif extract_type == "inclusion":
        save_path = os.path.join(
            save_path,
            f"{experiment.nct_id}",
            f"{model_cfg.model.replace('/', '-')}_inclusion_probs.csv",
        )
        to_enum = ["inclusion"]
    else:
        return input_df

    if os.path.exists(save_path):
        return pd.read_csv(save_path, index_col=0)

    # validate model_cfg
    assert model_cfg.get("model_type") == "text", "Model type must be 'text'."
    assert model_cfg.get("prompt_logprobs") == 0, "Prompt logprobs must be 0."

    model = LM(**model_cfg)

    input_df = experiment.discretize(input_df, hard_filter=False, inf=True)

    system_prompt = experiment.get_prompt("conditionals")

    options = enumerate_strings(experiment.get_options(to_enum))
    interleaved_options = qa_interleaved_enum(
        experiment.get_question_prompt(to_enum),
        experiment.get_options(to_enum),
        options,
        to_enum,
    )
    idx_to_feat = enum_to_dcts(options, to_enum)
    idx_to_feat = [experiment.transform_samples(dct) for dct in idx_to_feat]

    llm_probs_df = pd.DataFrame()

    for start in tqdm(range(0, len(input_df), batch_size)):
        batch_df = input_df.iloc[start : start + batch_size].reset_index(drop=True)

        reports = batch_df["report"].tolist()
        if extract_type != "inclusion":
            for idx, report in enumerate(reports):
                row = input_df.loc[input_df["report"] == report]
                if len(row) == 0:
                    continue
                to_sample = (
                    experiment.covariate_names
                    if extract_type == "ty_given_x"
                    else experiment.covariate_names + ["treatment"]
                )
                row = row[to_sample].to_dict("records")[0]
                sample_text = get_sample_text(row, experiment)
                reports[idx] += sample_text

        reports_repeated = [
            report for report in reports for _ in range(len(interleaved_options))
        ]
        options_repeated = interleaved_options * len(reports)
        llm_inputs = [
            report + option
            for report, option in zip(reports_repeated, options_repeated)
        ]

        cols = (
            experiment.covariate_names
            + experiment.outcome_names
            + ["treatment", "report"]
        )
        rows = batch_df[cols]

        lm_responses = [
            model.predict(prompt=system_prompt + "\n\n" + llm_input)
            for llm_input in llm_inputs
        ]

        logprobs = []
        for lm_response in lm_responses:
            logprob = sum(lm_response[0]["prompt_logprobs"])
            if length_norm:
                logprob = logprob / len(lm_response[0]["prompt_tokens"])
            logprobs.append(logprob)

        probs = softmax(
            np.array(logprobs).reshape((len(reports), len(interleaved_options))),
            axis=1,
        )
        sample_indices = [np.random.choice(len(prob), p=prob) for prob in probs]

        dict_to_save = [
            {
                **rows.iloc[j].to_dict(),
                **idx_to_feat[sample_indices[j]],
                **{"probs": probs[j]},
            }
            for j in range(len(reports))
        ]

        df_to_save = pd.DataFrame.from_dict(dict_to_save)
        llm_probs_df = pd.concat([llm_probs_df, df_to_save], ignore_index=True)
        llm_probs_df.to_csv(save_path)
        llm_inputs, rows = [], []

    llm_probs_df.to_csv(save_path)
    return pd.read_csv(save_path, index_col=0)


def weight_by_inclusion(ites, inclusion_probs):
    # ites has shape [num_treatments, num_datapoints]
    probs = inclusion_probs.apply(
        lambda row: [float(prob) for prob in row["probs"][1:-1].split()][1], axis=1
    ).to_numpy()
    return np.average(ites, axis=1, weights=probs)


@hydra.main(config_path="conf/", config_name="config.yaml", version_base="1.2")
def main(cfg: DictConfig) -> None:  # noqa: PLR0915
    experiment = SvT(path_to_main=cfg.user.path_to_main)
    os.makedirs(os.path.join(cfg.save_path, f"{experiment.nct_id}"), exist_ok=True)

    nest_asyncio.apply()

    data_flow = {}

    curated_df = pd.read_csv(experiment.curated_data_path, index_col=0)  # .head(10)
    data_flow["curated"] = len(curated_df)
    logger.info(f"Initial number of curated reports: {len(curated_df)} reports.")

    # filter reports that do not contain t,y info
    ty_samples = asyncio.run(
        extract_covariates(
            curated_df,
            experiment,
            cfg.cheap_model,
            cfg.save_path,
            "ty_filter",
            response_format=TYFilterResponse,
            api_calls_per_second=cfg.lm_api_calls_per_second,
            save_freq=cfg.extract_save_freq,
        )
    )
    ty_filtered_df = experiment.hard_filter_ty(ty_samples)
    data_flow["ty_filtered"] = len(ty_filtered_df)
    logger.info(f"After treatment-outcome filter: {len(ty_filtered_df)} reports.")

    # extract samples from reports, allowing LLM to output "unknown" for missing info
    samples_with_unknown = asyncio.run(
        extract_covariates(
            ty_filtered_df,
            experiment,
            cfg.sample_model,
            cfg.save_path,
            "knowns",
            response_format=KnownsResponse,
            api_calls_per_second=cfg.lm_api_calls_per_second,
            save_freq=cfg.extract_save_freq,
        )
    )

    # filter reports known to violate inclusion criteria
    inclusion_filtered = experiment.discretize(
        samples_with_unknown, hard_filter=True, inf=False
    )
    data_flow["inclusion_filtered"] = len(inclusion_filtered)
    logger.info(f"After inclusion filter: {len(inclusion_filtered)} reports.")

    # impute samples from reports, imputing missing info
    imputed_samples = asyncio.run(
        extract_covariates(
            inclusion_filtered,
            experiment,
            cfg.sample_model,
            cfg.save_path,
            "imputations",
            response_format=ImputationsResponse,
            api_calls_per_second=cfg.lm_api_calls_per_second,
            save_freq=cfg.extract_save_freq,
        )
    )

    # drop rows with missing covariates even after imputation
    imputed_samples = imputed_samples.dropna(
        subset=experiment.covariate_names
    ).reset_index(drop=True)
    data_flow["final"] = len(imputed_samples)
    logger.info(f"Final: {len(imputed_samples)} reports.")

    # extract conditionals depending on the estimator type
    estimator_type = cfg.estimator._target_.split(".")[-1]
    extract_type = {
        "NaturalIPW": "ty_given_x",
        "NaturalOI": "y_given_tx",
        "NaturalMC": None,
    }[estimator_type]
    conditionals = extract_conditionals(
        imputed_samples,
        experiment,
        cfg.probs_model,
        cfg.save_path,
        extract_type=extract_type,
    )

    # extract inclusion probabilities of the form P(X in I | R)
    inclusion_probs = extract_conditionals(
        imputed_samples,
        experiment,
        cfg.probs_model,
        cfg.save_path,
        extract_type="inclusion",
    )

    estimator = instantiate(cfg.estimator, experiment=experiment)
    result_dicts = []
    for outcome in experiment.outcome_names:
        all_ites = estimator.get_ites(conditionals, outcome)
        weighted_effects = weight_by_inclusion(
            all_ites, inclusion_probs
        )  # len: num_treatments

        for i, treat1 in enumerate(experiment.treatment_names):
            for j, treat2 in enumerate(experiment.treatment_names):
                if i < j:
                    pred_ate = weighted_effects[j] - weighted_effects[i]
                    results = {
                        "estimator": cfg.estimator._target_.split(".")[-1],
                        "outcome": outcome,
                        "treatments": f"{treat2}-{treat1}",
                        "pred_ate": pred_ate,
                    }
                    logger.info(f"Predicted ATE: {pred_ate}")
                    if experiment.split != "test":
                        effect_idx = experiment.outcome_treatment.index(
                            (outcome, (treat1, treat2))
                        )
                        true_ate = experiment.effect_sizes[effect_idx]
                        error = abs(pred_ate - true_ate)
                        results.update({"true_ate": true_ate, "abs_error": error})
                        logger.info(f"True ATE: {true_ate}")
                        logger.info(f"Absolute Error: {error}")
                    results.update(data_flow)
                    result_dicts.append(results)

    # TODO later: compute other evaluation metrics, e.g. sensitivity, balance

    result_df = pd.DataFrame(result_dicts)
    results_path = os.path.join(cfg.save_path, f"{experiment.nct_id}/ate_results.csv")
    if os.path.exists(results_path):
        existing_df = pd.read_csv(results_path, index_col=0)
        result_df = pd.concat([existing_df, result_df], ignore_index=True)
    result_df.to_csv(results_path)


if __name__ == "__main__":
    main()
