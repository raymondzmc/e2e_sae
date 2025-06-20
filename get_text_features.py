import heapq
import json
import os
import math
import random
from collections import defaultdict
from scipy.sparse import lil_matrix
from typing import Any

import torch
import wandb
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoTokenizer
import numpy as np

import asyncio # For running async explainer
from neuron_explainer.explanations.explainer import TokenActivationPairExplainer, PromptFormat # ContextSize if needed
from neuron_explainer.activations.activations import NeuronRecord, ActivationRecord
from neuron_explainer.activations.activation_records import calculate_max_activation
from neuron_explainer.explanations.calibrated_simulator import UncalibratedNeuronSimulator
from neuron_explainer.explanations.simulator import LogprobFreeExplanationTokenSimulator
from neuron_explainer.explanations.few_shot_examples import FewShotExampleSet
from neuron_explainer.explanations.scoring import simulate_and_score

from e2e_sae import SAETransformer
from e2e_sae.data import DatasetConfig, create_data_loader
from e2e_sae.scripts.analysis.utils import create_run_df

WINDOW_SIZE = 64
TARGET_FEATURES = 150
TOP_K = 100
NUM_FEATURES_FOR_EXPLANATION = 20
MIN_COUNT = 20
OPENAI_API_KEY = "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"
N_SAMPLES = 50000
OPENAI_MODELS = {
    "gpt-4o-mini": "gpt-4o-mini-07-18",
    "gpt-4o": "gpt-4o-2024-11-20",
}

def query_openai(messages: list[dict[str, str]], model: str = "gpt-4o-mini", **kwargs: Any) -> str:
    client = OpenAI(api_key=OPENAI_API_KEY)
    model_name = OPENAI_MODELS[model]
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.0,
        **kwargs
    )
    return response.choices[0].message.content.strip()


async def _run_simulation_and_scoring(
    explanation_text: str,
    records_for_simulation: list[ActivationRecord],
    model_name_for_simulator: str,
    few_shot_example_set: FewShotExampleSet,
    prompt_format: PromptFormat
) -> tuple[float | None, Any | None]:
    """Helper to run simulation and scoring with retries."""
    _score = None
    _scored_simulation_details = None
    for i in range(5):  # Retry loop
        try:
            print(f"Attempt {i+1}/5 to simulate and score...")
            simulator = UncalibratedNeuronSimulator(
                LogprobFreeExplanationTokenSimulator(
                    model_name_for_simulator,
                    explanation_text,
                    json_mode=True,
                    max_concurrent=10,
                    few_shot_example_set=few_shot_example_set,
                    prompt_format=prompt_format,
                )
            )
            _scored_simulation_details = await simulate_and_score(simulator, records_for_simulation)
            
            if _scored_simulation_details:
                _score = _scored_simulation_details.get_preferred_score()
                if _score is not None and np.isnan(_score): 
                    print(f"Warning: NaN score detected on attempt {i+1}. Inspecting aggregated activations from ScoredSimulation:")
                    if hasattr(_scored_simulation_details, 'scored_sequence_simulations'):
                        all_true_activations_debug = []
                        all_predicted_activations_debug = []
                        for seq_sim_detail in _scored_simulation_details.scored_sequence_simulations:
                            # seq_sim_detail is a ScoredSequenceSimulation object
                            if seq_sim_detail.true_activations is not None: # Accessing direct attribute
                                all_true_activations_debug.extend(seq_sim_detail.true_activations)
                            # Accessing predicted_activations via property (which in turn accesses simulation.expected_activations)
                            if hasattr(seq_sim_detail, 'predicted_activations') and seq_sim_detail.predicted_activations is not None:
                                all_predicted_activations_debug.extend(seq_sim_detail.predicted_activations)
                            elif seq_sim_detail.simulation and seq_sim_detail.simulation.expected_activations is not None:
                                # Fallback if property access was tricky, directly use underlying structure
                                all_predicted_activations_debug.extend(seq_sim_detail.simulation.expected_activations)

                        if not all_true_activations_debug:
                            print("  Collected all_true_activations_debug is EMPTY or all sequences had None.")
                        else:
                            print(f"  Collected all_true_activations_debug (sample of first 20): {all_true_activations_debug[:20]}")
                            true_std_debug = np.std(all_true_activations_debug)
                            print(f"  Std dev of all_true_activations_debug: {true_std_debug}")
                            if true_std_debug == 0:
                                print("  ISSUE: Standard deviation of ALL TRUE ACTIVATIONS is ZERO.")

                        if not all_predicted_activations_debug:
                            print("  Collected all_predicted_activations_debug is EMPTY or all sequences had None.")
                        else:
                            print(f"  Collected all_predicted_activations_debug (sample of first 20): {all_predicted_activations_debug[:20]}")
                            pred_std_debug = np.std(all_predicted_activations_debug)
                            print(f"  Std dev of all_predicted_activations_debug: {pred_std_debug}")
                            if pred_std_debug == 0:
                                print("  ISSUE: Standard deviation of ALL PREDICTED ACTIVATIONS is ZERO.")
                        
                        if not all_true_activations_debug or not all_predicted_activations_debug:
                             print("  One or both of the aggregated activation lists for correlation was empty.")

                    else:
                        print("  _scored_simulation_details does not have 'scored_sequence_simulations' or it is empty.")
            else:
                _score = None
                print(f"Simulation attempt {i+1} returned no details.")

        except Exception as e:
            print(f"ERROR during simulation attempt {i+1}, RETRYING: {e}")
        else:
            if _score is not None and not np.isnan(_score):
                break
            elif i == 4 and (_score is None or np.isnan(_score)):
                 print(f"Failed to get a valid score after 5 attempts. Last score: {_score}")
    return _score, _scored_simulation_details


device = torch.device('cuda')
api = wandb.Api(api_key='b8fa6d3104a0f99ee8a99f7c7659b893559f1097')
project = "raymondl/tinystories-1m_play"
runs = api.runs(project)
print(runs)
RUN_PATH = "raymondl/gpt2-local/l057k686"
runs = [api.run(RUN_PATH)]
activation_locations = [
    'blocks.2.hook_resid_pre',
    'blocks.4.hook_resid_pre',
    'blocks.6.hook_resid_pre'
]
for run in runs:
    run_id = run.id
    config = run.config
    config['eval_data']['streaming'] = False
    batch_size = config['batch_size']
    eval_config = DatasetConfig(**config['eval_data'])
    eval_loader = create_data_loader(
        eval_config, batch_size=batch_size, global_seed=config['seed']
    )[0]
    n_batches = math.ceil(N_SAMPLES / batch_size / (eval_config.n_ctx / WINDOW_SIZE))
    tokenizer = AutoTokenizer.from_pretrained(eval_config.tokenizer_name)
    neuron_activations_file = f"interpretability_results/neuron_activations_{run_id}.pt"
    neuron_activation_tokens_file = f"interpretability_results/neuron_activation_tokens_{run_id}.pt"

    if True:
        print(f"Obtaining features for {run_id}")
        model = SAETransformer.from_wandb(RUN_PATH).to(device)
        model.saes.eval()
        total_tokens = 0
        neuron_activations: dict[str, dict[int, list[tuple[int, list[int]]]]] = {}
        neuron_activation_tokens: list[list[str]] = []

        for sae_pos in model.raw_sae_positions:
            neuron_activations[sae_pos] = defaultdict(list)

        cache_positions: list[str] | None = None
        if config['loss']['in_to_orig'] is not None:
            assert set(config['loss']['in_to_orig']['hook_positions']).issubset(
                set(model.tlens_model.hook_dict.keys())
            ), "Some hook_positions in config.loss.in_to_orig.hook_positions are not in the model."
            # Don't add a cache position if there is already an SAE at that position which will cache
            # the inputs anyway
            cache_positions = [
                pos for pos in config.loss.in_to_orig.hook_positions if pos not in model.raw_sae_positions
            ]
        
        # Compute max activations for each neuron
        max_activations = {pos: None for pos in model.raw_sae_positions}
        for batch_idx, batch in tqdm(enumerate(eval_loader), total=n_batches, desc="Computing max activations"):
            if batch_idx >= n_batches:
                break
            token_ids = batch[eval_config.column_name].to(device)
            batch_size_, seq_len = token_ids.shape
            num_chunks = seq_len // WINDOW_SIZE
            token_ids_chunked = token_ids.view(batch_size_, num_chunks, WINDOW_SIZE)
            token_ids_chunked = token_ids_chunked.reshape(-1, WINDOW_SIZE)
            with torch.no_grad():
                _, new_acts = model.forward(
                    tokens=token_ids_chunked,
                    sae_positions=model.raw_sae_positions,
                    cache_positions=None,
                )
            for pos in model.raw_sae_positions:
                acts = new_acts[pos].c  # shape: [chunked_batch, seq_len, num_features]
                max_acts = acts.view(-1, acts.size(-1)).max(dim=0).values
                if max_activations[pos] is None:
                    max_activations[pos] = max_acts
                else:
                    max_activations[pos] = torch.maximum(max_activations[pos], max_acts)


        # Store activations for each neuron
        for batch_idx, batch in tqdm(enumerate(eval_loader), total=n_batches, desc="Eval Steps"):
            if batch_idx >= n_batches:
                break
            token_ids = batch[eval_config.column_name].to(device)
            total_tokens += token_ids.shape[0] * token_ids.shape[1]

            # Reshape token_ids to break into chunks of WINDOW_SIZE
            batch_size, seq_len = token_ids.shape
            if seq_len % WINDOW_SIZE != 0:
                raise ValueError(f"Sequence length {seq_len} is not divisible by WINDOW_SIZE {WINDOW_SIZE}")
            
            num_chunks = seq_len // WINDOW_SIZE
            chunked_batch_size = batch_size * num_chunks
            token_ids_chunked = token_ids.view(batch_size, num_chunks, WINDOW_SIZE)
            token_ids_chunked = token_ids_chunked.reshape(chunked_batch_size, WINDOW_SIZE)

            # Run through the SAE-augmented model
            with torch.no_grad():
                new_logits, new_acts = model.forward(
                    tokens=token_ids_chunked,
                    sae_positions=model.raw_sae_positions,
                    cache_positions=None,
                )

            for sae_pos in model.raw_sae_positions:
                acts = new_acts[sae_pos].c # chunked_batch_size x seq_len x num_dictionary_elements
                max_acts = max_activations[sae_pos]
                safe_max = torch.where(max_acts > 0, max_acts, torch.ones_like(max_acts))
                discretized_acts = torch.round((acts / safe_max.unsqueeze(0).unsqueeze(0)) * 10).to(torch.int8).cpu()
                nonzero_indices = discretized_acts.sum(1).nonzero(as_tuple=False).cpu()
                
                import pdb; pdb.set_trace()
                for data_idx, neuron_idx in nonzero_indices.tolist():
                    neuron_activations[sae_pos][neuron_idx].append((
                        len(neuron_activation_tokens) + data_idx,
                        discretized_acts[data_idx, :, neuron_idx].tolist()
                    ))
            chunked_tokens = [tokenizer.convert_ids_to_tokens(token_ids_chunked[i]) for i in range(chunked_batch_size)]
            neuron_activation_tokens.extend(chunked_tokens)

    #  # Initialize TokenActivationPairExplainer
    # explainer = TokenActivationPairExplainer(
    #     model_name=OPENAI_MODELS.get("gpt-4o", "gpt-4o"), # Use defined model name
    #     prompt_format=PromptFormat.HARMONY_V4
    # )

    # num_features = feature_counts.shape[0]

    # # Sort all features by their activation counts in descending order
    # sorted_indices_by_count = torch.argsort(feature_counts, descending=True)

    # # Select the indices of the top TARGET_FEATURES (e.g., 150)
    # candidate_top_feature_indices = sorted_indices_by_count[:TARGET_FEATURES]

    # # Filter these candidate top features by MIN_COUNT
    # # f_idx here will be a tensor element, so use .item() for Python int
    # features_to_explain_indices = [
    #     f_idx.item() for f_idx in candidate_top_feature_indices if feature_counts[f_idx].item() >= MIN_COUNT
    # ]
    
    # explanations_for_run = {}
    # print(f"\nStarting explanation generation for {len(features_to_explain_indices)} features (selected from top {TARGET_FEATURES}, filtered by MIN_COUNT >= {MIN_COUNT}) in run {run_id}...")

    # # Loop over the selected and filtered feature indices
    # for f_idx in features_to_explain_indices: # f_idx is now a Python int from the list
    #     # The MIN_COUNT check `if feature_counts[f_idx].item() < MIN_COUNT:`
    #     # is no longer needed here as it's handled during the construction of features_to_explain_indices.
        
    #     # print(f"Processing feature {f_idx} for explanation (count: {feature_counts[f_idx].item()})...")
        
    #     top_k_examples_for_feature_heap = feature_top[f_idx]
    #     if not top_k_examples_for_feature_heap:
    #         # print(f"Skipping feature {f_idx} for run {run_id} as it has no top examples.")
    #         continue

    #     # Get the actual top K examples (heapq stores a min-heap, so nlargest is appropriate)
    #     # Each item in heap: (max_act_val, window_token_ids_list, activations_list_for_feature)
    #     top_k_examples_for_feature = heapq.nlargest(NUM_FEATURES_FOR_EXPLANATION, top_k_examples_for_feature_heap)

    #     all_activation_records_for_feature = []
    #     for max_act_val, window_token_ids, activations_list_for_feature in top_k_examples_for_feature:
    #         # window_token_ids is already a list of ints
    #         # activations_list_for_feature is already a list of floats
            
    #         # Convert token IDs to token strings using the tokenizer
    #         # Using convert_ids_to_tokens which gives raw tokens (e.g. 'Ġword')
    #         # which is often what models like to see.
    #         tokens_as_strings = tokenizer.convert_ids_to_tokens(window_token_ids)
            
    #         # Filter out empty or problematic records if any
    #         if not tokens_as_strings or len(tokens_as_strings) != len(activations_list_for_feature):
    #             # print(f"Warning: Skipping malformed record for feature {f_idx}")
    #             continue

    #         activation_record = ActivationRecord(
    #             tokens=tokens_as_strings, 
    #             activations=activations_list_for_feature
    #         )
    #         all_activation_records_for_feature.append(activation_record)

    #     if not all_activation_records_for_feature:
    #         # print(f"Skipping feature {f_idx} for run {run_id} as no valid activation records could be formed after filtering.")
    #         continue
            
    #     max_overall_activation_for_feature = calculate_max_activation(all_activation_records_for_feature)
    #     if max_overall_activation_for_feature == 0: # Or some very small epsilon
    #          # print(f"Skipping feature {f_idx} for run {run_id} as max_activation is zero.")
    #          continue


    #     print(f"Generating explanation for feature {f_idx} (run {run_id}, {len(all_activation_records_for_feature)} examples)...")
    #     try:
    #         # generate_explanations is async, so use asyncio.run()
    #         generated_explanations = asyncio.run(explainer.generate_explanations(
    #             all_activation_records=all_activation_records_for_feature,
    #             max_activation=max_overall_activation_for_feature,
    #             num_samples=1, # Number of explanation candidates to generate
    #             max_tokens=100, # Max length of the explanation itself
    #             temperature=0.0 # Adjust for creativity if desired, 0.0 for deterministic
    #         ))
            
    #         if generated_explanations:
    #             explanation = generated_explanations[0] # Take the first one
    #             print(f"  Feature {f_idx} (Run {run_id}): {explanation}")
    #             # explanations_for_run[f_idx] = explanation # Will be updated later
    #         else:
    #             explanation = None # Ensure explanation is defined
    #             print(f"  No explanation generated for feature {f_idx} (Run {run_id}).")

    #     except Exception as e:
    #         print(f"  Error generating explanation for feature {f_idx} (Run {run_id}): {e}")
    #         explanation = None # Ensure explanation is defined in case of error during generation
    #         # explanations_for_run[f_idx] = {"text": f"Error in explanation: {e}", "score": None} # Store error
    #         # continue # Skip scoring if explanation failed
        
    #     temp_activation_records = [
    #         ActivationRecord(
    #             tokens=[
    #                 token.replace("<|endoftext|>", "<|not_endoftext|>"
    #                 )
    #                 .replace(" 55", "_55")
    #                 .encode("ascii", errors="backslashreplace")
    #                 .decode("ascii")
    #                 for token in activation_record.tokens
    #             ],
    #             activations=activation_record.activations,
    #         )
    #         for activation_record in all_activation_records_for_feature
    #     ]

    #     score = None
    #     scored_simulation_details = None # Use a different name to avoid conflict if one existed

    #     if explanation and temp_activation_records: # Only attempt scoring if we have an explanation and records
    #         # simulator_model_name = OPENAI_MODELS.get("gpt-4o-mini", "gpt-4o-mini")
    #         # Call the async helper using asyncio.run()
    #         score, scored_simulation_details = asyncio.run(
    #             _run_simulation_and_scoring(
    #                 explanation_text=explanation,
    #                 records_for_simulation=temp_activation_records,
    #                 model_name_for_simulator='gpt-4o-mini',
    #                 few_shot_example_set=FewShotExampleSet.JL_FINE_TUNED,
    #                 prompt_format=PromptFormat.HARMONY_V4
    #             )
    #         )

    #         if (
    #             score is None
    #             or scored_simulation_details is None
    #             or len(scored_simulation_details.scored_sequence_simulations) != len(temp_activation_records)
    #         ):
    #             print(
    #                 f"ERROR: Failed to properly score feature {f_idx} for run {run_id}. Score/Simulation invalid or mismatched. Score set to None."
    #             )
    #             score = None # Ensure score is None if checks fail
    #         else:
    #             print(f"  Feature {f_idx} (Run {run_id}) - Score: {score}")
    #     elif not explanation:
    #         print(f"  Skipping scoring for feature {f_idx} (Run {run_id}) as no explanation was generated.")
    #     elif not temp_activation_records:
    #          print(f"  Skipping scoring for feature {f_idx} (Run {run_id}) as no activation records were available for simulation.")

    #     # Update storage for explanations_for_run
    #     if explanation: 
    #         explanations_for_run[f_idx] = {"text": explanation, "score": score}
    #     elif f_idx not in explanations_for_run: # If explanation failed initially and wasn't stored
    #          explanations_for_run[f_idx] = {"text": "No explanation generated or error during generation", "score": None}
    #     import pdb; pdb.set_trace()
    # # Save all explanations for the current run to a JSON file
    # output_explanation_file = f"interpretability_results/explanations_{run_id}.json"
    # os.makedirs(os.path.dirname(output_explanation_file), exist_ok=True) # Ensure directory exists
    # with open(output_explanation_file, "w") as f:
    #     json.dump(explanations_for_run, f, indent=2)
    # print(f"Saved explanations for run {run_id} to {output_explanation_file}\n")