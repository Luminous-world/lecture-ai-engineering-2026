# app/ask.py
# Interactive CLI tool for multi-turn conversation with dynamic model switching, error rollback, and token count display.

"""第1回: LLMと会話履歴を保持しながら対話するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると対話モードが直接起動
"""

# section: 1
# description: Import required libraries, define constants, and add dynamic model fetching helper with generic filtering.
import argparse
import re
import sys
from pathlib import Path

from google import genai
from google.genai import errors

from app.config import MODEL as DEFAULT_CONFIG_MODEL, api_error_message, get_api_key

GLOBAL_VERSION = "1.8.0"  # 旧値: "1.7.0"

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"

DEFAULT_MODELS = (
    "gemini-3.8-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
)

def _parse_model_info(name: str) -> tuple[tuple[int, ...], int]:
    """Extract version tuple and tier priority (1: flash, 2: base, 3: pro)."""
    match = re.search(r"gemini-(\d+(?:\.\d+)?)", name)
    if match:
        v_str = match.group(1)
        v_tuple = tuple(int(x) for x in v_str.split("."))
    else:
        v_tuple = (0,)

    name_lower = name.lower()
    if "flash" in name_lower:
        tier_rank = 1
    elif "pro" in name_lower:
        tier_rank = 3
    else:
        tier_rank = 2

    return v_tuple, tier_rank


def fetch_available_models(client: genai.Client) -> list[str]:
    """Fetch available text models dynamically based purely on text capability, returning top 9 models."""
    print(f"[section: 1] version: {GLOBAL_VERSION}")
    try:
        raw_models = []
        # テキスト対話APIに適合しないメディア・音声用モデル種別のみ一般除外
        excluded_types = ("embed", "imagen", "tts", "live")

        for m in client.models.list():
            name = m.name.replace("models/", "") if hasattr(m, "name") and m.name else ""
            name_lower = name.lower()

            if "gemini" in name_lower and not any(t in name_lower for t in excluded_types):
                methods = getattr(m, "supported_generation_methods", []) or []
                if not methods or "generateContent" in methods:
                    raw_models.append(name)

        if not raw_models:
            raw_models = list(DEFAULT_MODELS)
    except Exception:
        raw_models = list(DEFAULT_MODELS)

    # バージョン降順、タイプ順（flash -> base -> pro）で全モデルをソート
    raw_models.sort(key=lambda m: (_parse_model_info(m)[0], -_parse_model_info(m)[1], m), reverse=True)
    # 上位9つのモデルのみを抽出
    return raw_models[:9]


# section: 2
# description: Interactive chat session with model selection and automatic error rollback.
def process_chat_message(chat, question: str) -> None:
    """Send a single prompt to the active chat session and display output with token counts."""
    response = chat.send_message(question)
    print(response.text)
    
    usage = response.usage_metadata
    if usage is not None:
        print(
            f"[トークン数] 入力: {usage.prompt_token_count}, "
            f"出力: {usage.candidates_token_count}"
        )


def select_model(client: genai.Client, current_model: str) -> str:
    """Display dynamically fetched model menu and return chosen model string."""
    print("\n========================================")
    print(" Fetching Available Models...")
    available_models = fetch_available_models(client)
    
    print(" Select Model Mode (Top 9 Models)")
    print("========================================")
    for idx, model_name in enumerate(available_models, start=1):
        marker = " (current)" if model_name == current_model else ""
        print(f"  [{idx}] {model_name}{marker}")
    print("  [0] Cancel (Keep current model)")
    print("========================================")

    choice = input(f"Select model number (0-{len(available_models)}): ").strip()
    if choice.isdigit():
        num = int(choice)
        if 1 <= num <= len(available_models):
            selected = available_models[num - 1]
            print(f"-> Switched model to: {selected}\n")
            return selected
    
    print("-> Model change cancelled.\n")
    return current_model
# [MODIFIED END]


def ask_interactive(initial_question: str | None = None) -> None:
    """Start interactive chat session supporting dynamic model switching with automatic rollback on error."""
    print(f"[section: 2] version: {GLOBAL_VERSION}")
    client = genai.Client(api_key=get_api_key())
    
    current_model = DEFAULT_CONFIG_MODEL
    chat = client.chats.create(model=current_model)

    print(f"[Active Model] {current_model}")

    if initial_question:
        print(f"[質問] {initial_question}\n")
        try:
            process_chat_message(chat, initial_question)
        except errors.APIError as e:
            print(api_error_message(e.code, e.message))

    while True:
        try:
            print("\n----------------------------------------")
            prompt_msg = f"[{current_model}] Enter question ('change model' to switch, 'exit' to quit): "
            user_input = input(prompt_msg).strip()
            
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("Ending conversation session. Goodbye!")
                break
            
            if user_input.lower() == "change model":
                new_model = select_model(client, current_model)
                if new_model != current_model:
                    history = chat.get_history()
                    prev_model = current_model
                    prev_chat = chat
                    
                    current_model = new_model
                    chat = client.chats.create(model=current_model, history=history)
                continue

            print(f"\n[質問] {user_input}\n")
            try:
                process_chat_message(chat, user_input)
            except errors.APIError as e:
                print(api_error_message(e.code, e.message))
                if 'prev_model' in locals() and prev_model != current_model:
                    print(f"-> Model '{current_model}' failed. Rolling back to '{prev_model}'.")
                    current_model = prev_model
                    chat = prev_chat
            
        except (KeyboardInterrupt, EOFError):
            print("\nSession terminated by user. Goodbye!")
            break


# section: 3
# description: Main entry point for command-line execution.
def main() -> None:
    print(f"[section: 3] version: {GLOBAL_VERSION}")
    parser = argparse.ArgumentParser(description="LLMと会話履歴を保って対話する")
    parser.add_argument("question", nargs="?", default=None, help="Initial question to start chat session")
    args = parser.parse_args()

    script_name = Path(sys.argv[0]).name
    print(f"[{script_name}] version: {GLOBAL_VERSION}")
    print(f"[model] {DEFAULT_CONFIG_MODEL}")

    try:
        ask_interactive(args.question)
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()