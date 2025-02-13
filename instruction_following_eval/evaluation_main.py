# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Binary of evaluating instruction following. See README.md."""

import collections
import dataclasses
import json
import os
from typing import Dict, Optional, Union

from absl import app
from absl import flags
from absl import logging

from instruction_following_eval import instructions_registry

# Define command-line flags.
_INPUT_DATA = flags.DEFINE_string(
    "input_data", None, "path to input data", required=True
)

_INPUT_RESPONSE_DATA = flags.DEFINE_string(
    "input_response_data", None, "path to input response data", required=False
)

_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir",
    None,
    "Output directory for inference and eval results.",
    required=True,
)


@dataclasses.dataclass
class InputExample:
  key: int
  instruction_id_list: list[str]
  prompt: str
  kwargs: list[Dict[str, Optional[Union[str, int]]]]


@dataclasses.dataclass
class OutputExample:
  instruction_id_list: list[str]
  prompt: str
  response: str
  follow_all_instructions: bool
  follow_instruction_list: list[bool]


def read_prompt_list(input_jsonl_filename):
  """Read inputs from jsonl."""
  inputs = []
  with open(input_jsonl_filename, "r") as f:
    for l in f:
      example = json.loads(l)
      inputs.append(
          InputExample(key=example["key"],
                       instruction_id_list=example["instruction_id_list"],
                       prompt=example["prompt"],
                       kwargs=example["kwargs"]))
  return inputs


def write_outputs(output_jsonl_filename, outputs):
  """Writes outputs to jsonl."""
  assert outputs
  with open(output_jsonl_filename, "w") as f:
    for o in outputs:
      f.write(
          json.dumps(
              {
                  attr_name: getattr(o, attr_name)
                  for attr_name in [name for name in dir(o) if not name.startswith("_")]
              }
          )
      )
      f.write("\n")


def normalize_prompt(prompt):
  """Helper function to ensure prompt keys match consistently."""
  if isinstance(prompt, list):
    return " ".join(prompt).strip()
  return prompt.strip()


def generate_responses(inputs):
    """
    Generates responses by sending prompts to the SGLang server.
    Assumes the SGLang API endpoint is running on http://localhost:5000/generate.
    """
    import requests
    api_url = "http://localhost:5000/generate"
    prompt_to_response = {}
    
    for inp in inputs:
        prompt_text = " ".join(inp.prompt) if isinstance(inp.prompt, list) else inp.prompt
        payload = {
            "text": prompt_text,  # Use key "text" as required by SGLang.
            "sampling_params": {
                "temperature": 0.6,
                "max_new_tokens": 256,  # Adjust as needed.
                "top_p": 0.95
            }
        }
        try:
            response = requests.post(api_url, json=payload)
            response.raise_for_status()
            data = response.json()
            # Use the correct key ("text") to retrieve the generated text.
            generated_response = data.get("text", "")
        except Exception as e:
            generated_response = ""
            print(f"Error generating response for prompt: {prompt_text}\nError: {e}")
            
        prompt_to_response[prompt_text.strip()] = generated_response.strip()
    return prompt_to_response


def read_prompt_to_response_dict(input_jsonl_filename):
  """Creates dictionary matching prompt and response."""
  return_dict = {}
  with open(input_jsonl_filename, "r") as f:
    for l in f:
      example = json.loads(l)
      # Normalize key as in generate_responses.
      key = normalize_prompt(example["prompt"])
      return_dict[key] = example["response"]
  return return_dict


def test_instruction_following_strict(inp, prompt_to_response):
  """Tests response to see if instructions are followed."""
  prompt_key = normalize_prompt(inp.prompt)
  response = prompt_to_response.get(prompt_key, "")
  instruction_list = inp.instruction_id_list
  is_following_list = []

  for index, instruction_id in enumerate(instruction_list):
    instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
    instruction = instruction_cls(instruction_id)
    instruction.build_description(**inp.kwargs[index])
    args = instruction.get_instruction_args()
    if args and "prompt" in args:
      instruction.build_description(prompt=inp.prompt)
    if response.strip() and instruction.check_following(response):
      is_following_list.append(True)
    else:
      is_following_list.append(False)

  return OutputExample(
      instruction_id_list=inp.instruction_id_list,
      prompt=inp.prompt,
      response=response,
      follow_all_instructions=all(is_following_list),
      follow_instruction_list=is_following_list,
  )


def test_instruction_following_loose(inp, prompt_to_response):
  """Tests response for an upper bound for following instructions."""
  prompt_key = normalize_prompt(inp.prompt)
  response = prompt_to_response.get(prompt_key, "")
  r = response.split("\n")
  response_remove_first = "\n".join(r[1:]).strip()
  response_remove_last = "\n".join(r[:-1]).strip()
  response_remove_both = "\n".join(r[1:-1]).strip()
  revised_response = response.replace("*", "")
  revised_response_remove_first = response_remove_first.replace("*", "")
  revised_response_remove_last = response_remove_last.replace("*", "")
  revised_response_remove_both = response_remove_both.replace("*", "")
  all_responses = [
      response,
      revised_response,
      response_remove_first,
      response_remove_last,
      response_remove_both,
      revised_response_remove_first,
      revised_response_remove_last,
      revised_response_remove_both,
  ]
  instruction_list = inp.instruction_id_list
  is_following_list = []
  for index, instruction_id in enumerate(instruction_list):
    instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
    instruction = instruction_cls(instruction_id)
    instruction.build_description(**inp.kwargs[index])
    args = instruction.get_instruction_args()
    if args and "prompt" in args:
      instruction.build_description(prompt=inp.prompt)
    is_following = any(r.strip() and instruction.check_following(r) for r in all_responses)
    is_following_list.append(is_following)
  return OutputExample(
      instruction_id_list=inp.instruction_id_list,
      prompt=inp.prompt,
      response=response,
      follow_all_instructions=all(is_following_list),
      follow_instruction_list=is_following_list,
  )


def print_report(outputs):
  """Prints a report on accuracy scores."""
  prompt_total = 0
  prompt_correct = 0
  instruction_total = 0
  instruction_correct = 0

  tier0_total = collections.defaultdict(int)
  tier0_correct = collections.defaultdict(int)
  tier1_total = collections.defaultdict(int)
  tier1_correct = collections.defaultdict(int)

  for example in outputs:
    follow_instruction_list = example.follow_instruction_list
    instruction_id_list = example.instruction_id_list

    prompt_total += 1
    if all(follow_instruction_list):
      prompt_correct += 1

    instruction_total += len(instruction_id_list)
    instruction_correct += sum(follow_instruction_list)

    for instruction_id, followed_or_not in zip(instruction_id_list, follow_instruction_list):
      tid = instruction_id.split(":")[0]
      tier0_total[tid] += 1
      if followed_or_not:
        tier0_correct[tid] += 1

    for instruction_id, followed_or_not in zip(instruction_id_list, follow_instruction_list):
      tier1_total[instruction_id] += 1
      if followed_or_not:
        tier1_correct[instruction_id] += 1

  print(f"prompt-level: {prompt_correct / prompt_total if prompt_total else 0.0}")
  print(f"instruction-level: {instruction_correct / instruction_total if instruction_total else 0.0}")
  print()
  for instruction_id in sorted(tier0_total.keys()):
    accuracy = tier0_correct[instruction_id] / tier0_total[instruction_id]
    print(f"{instruction_id} {accuracy}")
  print()
  for instruction_id in sorted(tier1_total.keys()):
    accuracy = tier1_correct[instruction_id] / tier1_total[instruction_id]
    print(f"{instruction_id} {accuracy}")


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  inputs = read_prompt_list(_INPUT_DATA.value)
  # If input_response_data is provided, load it; otherwise, use SGLang to generate responses.
  if _INPUT_RESPONSE_DATA.value:
    prompt_to_response = read_prompt_to_response_dict(_INPUT_RESPONSE_DATA.value)
  else:
    logging.info("No input response data provided; generating responses using SGLang model.")
    prompt_to_response = generate_responses(inputs)

  # Evaluate using both strict and loose criteria.
  for func, output_file_name in [
      (test_instruction_following_strict, "eval_results_strict"),
      (test_instruction_following_loose, "eval_results_loose"),
  ]:
    logging.info("Generating %s...", output_file_name)
    outputs = [func(inp, prompt_to_response) for inp in inputs]
    follow_all_instructions = [o.follow_all_instructions for o in outputs]
    accuracy = sum(follow_all_instructions) / len(outputs)
    logging.info("Accuracy: %f", accuracy)

    full_output_path = os.path.join(_OUTPUT_DIR.value, output_file_name + ".jsonl")
    write_outputs(full_output_path, outputs)
    logging.info("Generated: %s", full_output_path)

    # Prints instruction following accuracy report.
    print("=" * 64)
    print(f"{full_output_path} Accuracy Scores:")
    print_report(outputs)


if __name__ == "__main__":
  app.run(main)