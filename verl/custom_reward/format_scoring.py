from __future__ import annotations

import re
import string

from verl.prompts import THINK_PATTERN


def normalize_answer(value: str) -> str:
    lowered = value.lower()
    without_punctuation = "".join(character for character in lowered if character not in string.punctuation)
    without_articles = re.sub(r"\b(a|an|the)\b", " ", without_punctuation)
    return " ".join(without_articles.split())


def challenger_answer_reward(document: str, answer: str) -> float:
    normalized_answer = normalize_answer(answer)
    answer_word_count = len(normalized_answer.split())
    if normalized_answer not in {"yes", "no"} and normalized_answer not in normalize_answer(document):
        return 0.0
    if normalized_answer and answer_word_count <= 5:
        return 1.0
    if normalized_answer and answer_word_count <= 10:
        return 0.5
    return 0.0


def challenger_think_reward(assistant_messages: list[str]) -> float:
    matched_messages = sum(
        bool(re.match(THINK_PATTERN, message.strip(), re.DOTALL))
        for message in assistant_messages
    )
    return matched_messages / max(1, len(assistant_messages))


def challenger_format_score(
    *,
    question: str,
    answer: str,
    think_reward: float,
    tool_reward: float,
    answer_reward: float,
) -> float:
    has_valid_qa = (
        bool(question)
        and bool(answer)
        and normalize_answer(answer) not in normalize_answer(question)
    )
    if not has_valid_qa or tool_reward != 1.0:
        return 0.0
    return (1.0 + think_reward + tool_reward + answer_reward) / 4
