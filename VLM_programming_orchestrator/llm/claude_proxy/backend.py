"""LLM-based robot code generation using a Claude-compatible SSE proxy.

Note: `ABB_task_description_VLM.txt` and `ABB_Best_Practices.txt` live in the
LLM's RAG system — we reference them by name so the model retrieves them itself.
We do NOT inject their contents into the prompt.

The proxy emits Server-Sent Events of three kinds:
  - `status`  : intermediate "thinking" / function-call status updates
  - `token`   : streamed text tokens of the final answer
  - `done`    : terminal event; stream complete
  - `error`   : terminal event; stream failed
"""

import json
import os

import requests
# Disable SSL warnings for self-signed certificates
import urllib3
from requests_sse import (EventSource, InvalidContentTypeError,
                          InvalidStatusCodeError)

from ...config import Config
from ...vlm_stack_blocks.code_generation import build_stacking_prompt

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

LLM_BACKEND = "claude_proxy"


def _stream_llm_response(llm_api_url, messages, events=None):
    """Stream the LLM response over SSE and return the final text.

    `events` is an optional `PipelineEvents` instance; if supplied, intermediate
    `status` and `token` chunks are forwarded so the GUI can render the
    "thinking" phase live. Returns the accumulated token text on success, or
    `None` on failure.
    """
    payload = {"messages": messages}

    if events is not None:
        events.llm_stream_start()

    buffer_parts: list[str] = []
    success = False
    err_msg = ""

    try:
        with EventSource(
            llm_api_url,
            timeout=180,
            verify=False,
            headers={"Accept-Encoding": "identity"},
            method="POST",
            json=payload,
        ) as event_source:
            try:
                for event in event_source:
                    if event.type != "message":
                        continue
                    try:
                        data = json.loads(event.data)
                    except json.JSONDecodeError:
                        continue
                    kind = data.get("type")
                    content = data.get("content", "")
                    if kind == "status":
                        if events is not None:
                            events.llm_stream_chunk("status", content)
                    elif kind == "token":
                        buffer_parts.append(content)
                        if events is not None:
                            events.llm_stream_chunk("token", content)
                    elif kind == "error":
                        err_msg = content or "Unknown stream error"
                        print(f"\nLLM stream error: {err_msg}")
                        break
                    elif kind == "done":
                        success = True
                        break
            except InvalidStatusCodeError as e:
                err_msg = f"Invalid status code: {e}"
                print(err_msg)
            except InvalidContentTypeError as e:
                err_msg = f"Invalid content type: {e}"
                print(err_msg)
            except requests.RequestException as e:
                err_msg = f"Request error: {e}"
                print(err_msg)
    except Exception as e:  # pylint: disable=W0718
        err_msg = f"Failed to open SSE stream: {e}"
        print(err_msg)
    finally:
        if events is not None:
            events.llm_stream_end(success=success, message=err_msg)

    if not success:
        return None
    return "".join(buffer_parts)


def call_llm_for_robot_code(
    url,
    stacking_plan_3d,
    robot_type="ABB",
    code_language="Rapid",
    task_description_file="ABB_task_description_VLM.txt",
    best_practices_file="ABB_Best_Practices.txt",
    events=None,
):
    """Generate robot control code from a 3D stacking plan.

    Returns (generated_text, prompt). `prompt` can be passed back to
    `call_llm_to_fix_code` so the fix request carries the original context.
    `events` is forwarded to the SSE streamer for live GUI updates.
    """
    prompt = build_stacking_prompt(
        stacking_plan_3d,
        robot_type=robot_type,
        code_language=code_language,
        task_description_file=task_description_file,
        best_practices_file=best_practices_file,
    )

    print("\n=== Calling LLM for robot code generation ===")
    generated = _stream_llm_response(
        url, [{"role": "user", "content": prompt}], events=events
    )
    if generated is None:
        return None, prompt

    os.makedirs("VLM_output", exist_ok=True)
    output_path = os.path.join("VLM_output", "4_robot_control_code.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(generated)
    print(f"Generated robot code saved to: {output_path}")
    return generated, prompt


def call_llm_to_fix_code(
    url,
    current_code,
    error_message,
    original_prompt,
    robot_type="ABB",
    code_language="Rapid",
    events=None,
):
    """Ask Claude to fix the given code based on an error/feedback message.

    The original prompt is replayed as prior context so the model still knows
    about the RAG files and the stacking plan.
    """
    fix_message = (
        f"The {code_language} code you produced for the {robot_type} robot has a problem.\n\n"
        # f"```{code_language.lower()}\n{current_code}\n```\n\n"
        f"Problem / error message:\n{error_message}\n\n"
        f"Return the corrected full module only, inside a single code block."
    )

    messages = [
        {"role": "user", "content": original_prompt},
        {"role": "assistant", "content": current_code},
        {"role": "user", "content": fix_message},
    ]

    print("\n=== Calling Claude to fix robot code ===")
    fixed = _stream_llm_response(url, messages, events=events)
    if fixed is None:
        return None

    os.makedirs("VLM_output", exist_ok=True)
    output_path = os.path.join("VLM_output", "4_robot_control_code.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(fixed)
    print(f"Fixed robot code saved to: {output_path}")
    return fixed


class ClaudeProxyCodeGen:
    """`CodeGenBackend` implementation talking to a Claude-compatible SSE proxy."""

    def __init__(self, config: Config):
        self.config = config

    def generate_code(
        self,
        stacking_plan_3d,
        robot_type=None,
        code_language=None,
        task_description_file=None,
        best_practices_file=None,
        events=None,
    ):
        return call_llm_for_robot_code(
            self.config.base_url,
            stacking_plan_3d,
            robot_type=robot_type or self.config.robot_type,
            code_language=code_language or self.config.code_language,
            task_description_file=task_description_file or self.config.task_description_file,
            best_practices_file=best_practices_file or self.config.best_practices_file,
            events=events,
        )

    def fix_code(
        self,
        current_code,
        error_message,
        original_prompt,
        robot_type=None,
        code_language=None,
        events=None,
    ):
        return call_llm_to_fix_code(
            self.config.base_url,
            current_code,
            error_message,
            original_prompt,
            robot_type=robot_type or self.config.robot_type,
            code_language=code_language or self.config.code_language,
            events=events,
        )


def create_backend(config: Config) -> ClaudeProxyCodeGen:
    """Build the Claude-proxy `CodeGenBackend` for the given config."""
    return ClaudeProxyCodeGen(config)
