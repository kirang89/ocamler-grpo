import csv
import re


ASSERT_RE = re.compile(r"\bassert\b")
LET_RE = re.compile(r"^let(\s|$)")


def has_at_least_three_asserts(tests: str) -> bool:
    return len(ASSERT_RE.findall(tests)) >= 3


def normalize_prompt(prompt: str) -> str:
    # CSV stores literal \\n sequences for prompts; normalize to real newlines.
    return prompt.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\r", "\n")


def prompt_starts_with_comment(prompt: str) -> bool:
    normalized = normalize_prompt(prompt)
    return normalized.lstrip().startswith("(*")


def prompt_ends_with_let(prompt: str) -> bool:
    normalized = normalize_prompt(prompt)
    lines = [line for line in normalized.splitlines() if line.strip()]
    if not lines:
        return False
    return LET_RE.match(lines[-1].lstrip()) is not None


def main() -> None:
    with open("problems_2.csv", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row_id = row.get("id", "")
            prompt = row.get("prompt", "") or ""
            tests = row.get("tests", "") or ""

            failures = [
                not tests.strip(),
                not prompt_starts_with_comment(prompt),
                not prompt_ends_with_let(prompt),
                not has_at_least_three_asserts(tests),
            ]

            if any(failures):
                print(row_id)


if __name__ == "__main__":
    main()
