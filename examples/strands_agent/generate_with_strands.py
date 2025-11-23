# Adapted from https://github.com/volcengine/verl/blob/cb809d66e46dfd3342d008628891a14a054fa424/recipe/retool/retool.py

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.types import Sample

# Import reward models
try:
    from slime.rollout.rm_hub.math_dapo_utils import compute_score as math_dapo_compute_score
except ImportError:
    raise ImportError("MathDapo is not installed")

from strands import Agent
from strands.models.openai import OpenAIModel

from code_sandbox import CodeSandbox


async def generate(args, sample: Sample, sampling_params) -> Sample:
    """Generate function using strands-agents for tool calling"""
    assert not args.partial_rollout, "Partial rollout is not supported for this function at the moment."

    state = GenerateState(args)

    # Create OpenAI-compatible model pointing to sglang server
    model = OpenAIModel(
        client_args={
            "api_key": "EMPTY",
            "base_url": f"http://{args.sglang_router_ip}:{args.sglang_router_port}/v1",
        },
        model_id=args.model_name,
        params={
            "max_tokens": sampling_params.get("max_new_tokens", 2048),
            "temperature": sampling_params.get("temperature", 1.0),
            "top_p": sampling_params.get("top_p", 0.7),
        },
    )

    # Create code sandbox
    code_sandbox = CodeSandbox(
        workdir="./tmp/code_sandbox",
        language="python",
        execution_timeout=300,
    )
    code_sandbox.start_session()

    # Create agent with code sandbox tool
    tools_list = [code_sandbox.execute_python_code]
    
    # Custom system prompt to encourage frequent tool usage
    math_system_prompt = (
        "You are a helpful math solver assistant with access to a python code execution tool. "
        "For math problems, USE THE PYTHON CODE EXECUTION TOOL FREQUENTLY to verify calculations - "
        "do not try to solve complex equations mentally. "
        "Break down the problem into steps and use the python code execution tool for each numerical computation. "
        "Keep your thinking concise and focus on using tools to solve the problem."
    )
    
    agent = Agent(model=model, tools=tools_list, system_prompt=math_system_prompt)

    # Track tool calls
    tool_call_count = 0

    try:
        # Run the agent with the prompt
        agent(sample.prompt)

        # Get OpenAI-formatted messages from the agent
        openai_messages = agent.model.format_request_messages(
            messages=agent.messages, system_prompt=agent.system_prompt
        )
        
        # Debug: print message structure
        import os
        if os.environ.get("DEBUG_STRANDS"):
            print("\n" + "="*80)
            print("DEBUG: OpenAI Messages Structure")
            print("="*80)
            for i, msg in enumerate(openai_messages):
                print(f"\nMessage {i}: role={msg.get('role')}")
                if "tool_calls" in msg:
                    print(f"  Has {len(msg.get('tool_calls', []))} tool calls")
                if "content" in msg:
                    content = msg.get("content")
                    if isinstance(content, str):
                        print(f"  Content (first 200 chars): {content[:200]}")
                    else:
                        print(f"  Content type: {type(content)}")

        # Count tool calls from OpenAI-formatted messages (more reliable)
        # In OpenAI format, assistant messages have a "tool_calls" field when tools are used
        for message in openai_messages:
            if message.get("role") == "assistant" and "tool_calls" in message:
                tool_calls = message.get("tool_calls", [])
                if tool_calls:  # tool_calls is a list of tool call objects
                    tool_call_count += len(tool_calls)
        
        # Convert content from list[dict] format to string format for chat template
        # The strands library returns content as [{"type": "text", "text": "..."}]
        # but the tokenizer's chat template expects just the string
        for message in openai_messages:
            if "content" in message and isinstance(message["content"], list):
                if len(message["content"]) > 0 and "text" in message["content"][0]:
                    message["content"] = message["content"][0]["text"]
                else:
                    message["content"] = ""

        # Apply chat template progressively to maintain proper alignment
        # First, get the prompt (system + initial user message)
        initial_prompt_messages = [msg for msg in openai_messages if msg["role"] in ["system", "user"]][
            :2
        ]  # system + first user
        prompt_text = state.tokenizer.apply_chat_template(
            initial_prompt_messages,
            tokenize=False,
            add_generation_prompt=True,  # Add generation prompt for the assistant
        )
        prompt_tokens_ids = state.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

        # Apply chat template to the full conversation
        full_conversation = state.tokenizer.apply_chat_template(
            openai_messages, tokenize=False, add_generation_prompt=False
        )
        all_token_ids = state.tokenizer(full_conversation, add_special_tokens=False)["input_ids"]

        # Response tokens are everything after the prompt
        response_token_ids = all_token_ids[len(prompt_tokens_ids) :]

        # Create loss masks by progressively building up the conversation
        # to determine token boundaries for each message
        loss_masks = []

        # Start with the initial prompt messages
        current_messages = list(initial_prompt_messages)
        prev_token_count = len(prompt_tokens_ids)

        # Iterate through remaining messages (assistant and tool messages)
        for message in openai_messages[len(initial_prompt_messages) :]:
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

            # Mask: 1 for assistant messages (we train on these), 0 for tool results
            if message["role"] == "assistant":
                loss_masks.extend([1] * message_token_length)
            else:  # tool messages
                loss_masks.extend([0] * message_token_length)

            prev_token_count = new_token_count

        # Ensure loss_masks matches response_token_ids length
        if len(loss_masks) > len(response_token_ids):
            loss_masks = loss_masks[: len(response_token_ids)]
        elif len(loss_masks) < len(response_token_ids):
            # Pad with 1s if needed (shouldn't happen but safety check)
            loss_masks.extend([1] * (len(response_token_ids) - len(loss_masks)))

        # Extract just the response text for logging/storage
        # This is in chat template format with all tool call tokens - needed for training
        full_response = full_conversation[len(prompt_text) :]
        
        # Debug output
        if os.environ.get("DEBUG_STRANDS"):
            # Also extract human-readable assistant text for debugging
            assistant_texts = []
            for msg in openai_messages[len(initial_prompt_messages):]:
                if msg.get("role") == "assistant" and "content" in msg:
                    content = msg.get("content", "")
                    # Handle both string and list[dict] content formats
                    if isinstance(content, list):
                        text = content[0].get("text", "") if len(content) > 0 else ""
                    else:
                        text = content
                    if text.strip():
                        assistant_texts.append(text)
            
            print("\n" + "="*80)
            print("DEBUG: Chat Template Response (for training)")
            print("="*80)
            print(full_response[:500])
            print("\n" + "="*80)
            print("DEBUG: Human-Readable Assistant Text (for viewing)")
            print("="*80)
            for i, text in enumerate(assistant_texts):
                print(f"\nAssistant Response {i+1}:\n{text[:300]}")
            print("="*80)

        # Set sample attributes
        sample.tokens = prompt_tokens_ids + response_token_ids
        sample.response_length = len(response_token_ids)
        sample.response = full_response
        sample.loss_mask = loss_masks

        # Store information for wandb logging
        sample.payload_text = sample.prompt + full_response
        sample.payload_has_system = True  # strands uses system prompts
        sample.payload_has_tools = len(tools_list) > 0

        # Store tool call count for reward calculation
        sample.tool_call_count = tool_call_count

        # Set status as completed
        sample.status = Sample.Status.COMPLETED

        # Log to wandb if available
        try:
            import wandb

            if wandb.run is not None:
                wandb.log(
                    {
                        "debug/response_length": len(full_response),
                        "debug/available_tools": len(tools_list),
                        "debug/tool_calls": tool_call_count,
                        "debug/num_messages": len(agent.messages),
                    }
                )
        except ImportError:
            pass

    except Exception as e:
        # Handle any errors during generation
        sample.status = Sample.Status.ABORTED
        sample.response = f"Error: {str(e)}"
        sample.response_length = 0
        sample.tokens = state.tokenizer(sample.prompt, add_special_tokens=False)["input_ids"]
        sample.loss_mask = []
        sample.tool_call_count = 0

    finally:
        code_sandbox.close_session()

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

    # use \\boxed{...} answer
    result = math_dapo_compute_score(solution_str, ground_truth, strict_box_verify=True)

    # encourage model to call tools
    if result["score"] < 0:
        tool_call_reward = (num_turns - 2) / 2 * 0.1
        result["score"] = min(-0.6, result["score"] + tool_call_reward)

    if result["pred"] is None:
        result["pred"] = ""

    return result
