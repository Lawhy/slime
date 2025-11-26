import logging

from strands import Agent, tool
from strands.models.openai import OpenAIModel
from strands.types.exceptions import MaxTokensReachedException
import wandb

from examples.strands_agent.code_interpreter import CodeInterpreter
from slime.rollout.rm_hub.math_dapo_utils import (
    compute_score as math_dapo_compute_score,
)
from slime.rollout.sglang_rollout import GenerateState
from slime.utils.types import Sample

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """
You are a helpful assistant that can use Python tools to solve mathematical problems. When you need to perform calculations, use the `execute_code` tool to execute code and get results.
"""


def create_strands_agent(args, sampling_params) -> Agent:
    model = OpenAIModel(
        client_args={
            "api_key": "EMPTY",
            "base_url": f"http://{args.sglang_router_ip}:{args.sglang_router_port}/v1",
        },
        model_id=args.hf_checkpoint.split("/")[-1],
        params={
            "max_tokens": sampling_params["max_new_tokens"],
            "temperature": sampling_params["temperature"],
            "top_p": sampling_params["top_p"],
        },
    )

    @tool
    def execute_code(code: str) -> str:
        r"""Execute a given code snippet.

        Args:
            code (str): The input code to the Code Interpreter tool call.

        Returns:
            str: The text output from the Code Interpreter tool call.
        """
        code_interpreter = CodeInterpreter(require_confirm=False, execution_timeout=300)
        return code_interpreter.run(code=code, code_type="python")

    agent = Agent(model=model, tools=[execute_code], system_prompt=SYSTEM_PROMPT)
    return agent


def run_strands_agent(agent: Agent, prompt: str) -> Sample.Status:
    """Run the strands agent with the given prompt and set the sample status."""
    try:
        assert isinstance(prompt, str), "Prompt must be a string"
        logger.info(f"[Strands Agents] Running agent with prompt: {prompt}")
        agent(prompt=prompt)
        # Set status as completed
        sample_status = Sample.Status.COMPLETED
    except Exception as e:
        if isinstance(e, MaxTokensReachedException):
            sample_status = Sample.Status.TRUNCATED
        else:
            sample_status = Sample.Status.ABORTED
        logger.error(f"[Strands Agents] {e}")
    finally:
        pass

    return sample_status


def get_trajectory(agent: Agent) -> list[dict]:
    """Get the chat template-compatible trajectory of the strands agent."""
    trajectory = agent.model.format_request_messages(messages=agent.messages, system_prompt=agent.system_prompt)
    # Convert content from list[dict] format to string format for chat template
    # The strands library returns content as [{"type": "text", "text": "..."}]
    # but the tokenizer's chat template expects just the string
    for message in trajectory:
        if "content" in message and isinstance(message["content"], list):
            if len(message["content"]) > 0 and "text" in message["content"][0]:
                message["content"] = message["content"][0]["text"]
            else:
                message["content"] = ""
    return trajectory


async def generate(args, sample: Sample, sampling_params) -> Sample:
    """Generate function using strands-agents as agent scaffolding"""
    assert not args.partial_rollout, "Partial rollout is not supported for this function at the moment."

    state = GenerateState(args)

    # Create strands agent and run it with the sample prompt
    agent = create_strands_agent(args, sampling_params)
    logger.info(f"[Strands Agents] Sample prompt: {sample.prompt}, type: {type(sample.prompt)}")
    prompt_text = sample.prompt if isinstance(sample.prompt, str) else sample.prompt[0]["content"]
    sample.status = run_strands_agent(agent, prompt_text)
    trajectory = get_trajectory(agent)

    if sample.status == Sample.Status.ABORTED:
        return sample

    # Incremental tokenization approach (like retool)
    # Step 1: Get the initial prompt (system + user message)
    initial_prompt_messages = [msg for msg in trajectory if msg["role"] in ["system", "user"]][
        :2
    ]  # system + first user
    prompt_text = state.tokenizer.apply_chat_template(
        initial_prompt_messages,
        tokenize=False,
        add_generation_prompt=True,  # Add generation prompt for the assistant
    )
    prompt_tokens_ids = state.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

    # Step 2: Build response incrementally, tokenizing each message as we go
    response_token_ids = []
    loss_masks = []
    response_text = ""

    # Start with the initial prompt messages for progressive chat template application
    current_messages = list(initial_prompt_messages)
    prev_token_count = len(prompt_tokens_ids)

    # Iterate through remaining messages (assistant and tool messages)
    for message in trajectory[len(initial_prompt_messages) :]:
        # Add this message to the conversation
        current_messages.append(message)

        # Apply chat template and tokenize up to this point
        current_text = state.tokenizer.apply_chat_template(
            current_messages, tokenize=False, add_generation_prompt=False
        )
        current_token_ids = state.tokenizer(current_text, add_special_tokens=False)["input_ids"]

        # Calculate how many new tokens this message added
        new_token_count = len(current_token_ids)
        message_token_length = new_token_count - prev_token_count

        # Extract the new tokens for this message
        message_tokens = current_token_ids[prev_token_count:]
        response_token_ids.extend(message_tokens)

        # Mask: 1 for assistant messages (we train on these), 0 for tool results
        if message["role"] == "assistant":
            loss_masks.extend([1] * message_token_length)
        else:  # tool messages
            loss_masks.extend([0] * message_token_length)

        prev_token_count = new_token_count

    # Extract the response text (everything after the initial prompt)
    full_conversation_text = state.tokenizer.apply_chat_template(
        trajectory, tokenize=False, add_generation_prompt=False
    )
    response_text = full_conversation_text[len(prompt_text) :]

    # Set sample attributes
    sample.tokens = prompt_tokens_ids + response_token_ids
    sample.response_length = len(response_token_ids)
    sample.response = response_text
    sample.loss_mask = loss_masks

    # Store information for wandb logging
    sample.payload_text = sample.prompt + response_text
    sample.payload_has_system = True  # strands uses system prompts
    sample.payload_has_tools = len(agent.tool_names) > 0

    # Store tool call count for reward calculation
    sample.tool_call_count = [message["role"] == "tool" for message in trajectory].count(True)

    # Log to wandb if available
    if wandb.run is not None:
        wandb.log(
            {
                "debug/response_length": len(response_text),
                "debug/available_tools": len(agent.tool_names),
                "debug/tool_calls": sample.tool_call_count,
                "debug/num_messages": len(trajectory),
            }
        )

    return sample


async def reward_func(args, sample, **kwargs):
    """Tool call reward function using math_dapo as primary reward model"""
    if not isinstance(sample, Sample):
        raise TypeError("Sample must be an instance of Sample class.")

    # Build complete solution string
    solution_str = sample.prompt + sample.response

    # Get ground truth answer - label is a string, not a dict
    ground_truth = sample.label if sample.label is not None else ""

    # Get tool call count as num_turns
    num_turns = getattr(sample, "tool_call_count", 0)

    # Accept both Answer: ... and \\boxed{...} answer
    result = math_dapo_compute_score(solution_str, ground_truth, strict_box_verify=False)
    result_boxed = math_dapo_compute_score(solution_str, ground_truth, strict_box_verify=True)
    if result["pred"] == "[INVALID]":
        result = result_boxed

    # encourage model to call tools
    if result["score"] < 0:
        tool_call_reward = (num_turns - 2) / 2 * 0.1
        result["score"] = min(-0.6, result["score"] + tool_call_reward)

    if result["pred"] is None:
        result["pred"] = ""

    return result
