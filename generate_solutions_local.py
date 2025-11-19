import csv
import os
import re
import subprocess
import tempfile
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import requests

SYSTEM_PROMPT = "Respond only with runnable OCaml code (no prose)."

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:1.5b")
GENERATED_DIR = Path("generated_ocaml")

PROMPT_TEMPLATE = textwrap.dedent(
    """
    You are an expert OCaml engineer. Read the programming problem below and craft an OCaml
    solution plus a lightweight test harness that executes when run. Do not respond with
    any prose or text other than the code.

    Problem ({problem_id}):
    {question}
    """
).strip()


def call_ollama(prompt: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "system": SYSTEM_PROMPT,
        "stream": False,
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=300)
    response.raise_for_status()
    data = response.json()
    if "response" not in data:
        raise ValueError("Unexpected response from Ollama: missing 'response'")
    return data["response"].strip()


CODE_BLOCK_RE = re.compile(r"```(.*?)```", re.DOTALL)
LANGUAGE_HINTS = {"ocaml", "ml", "code", "language", "language:ocaml"}


def parse_response(raw_response: str) -> str:
    content = raw_response.strip()
    matches = CODE_BLOCK_RE.findall(content)
    if not matches:
        raise ValueError("No code block found in Ollama response")

    for block in matches:
        block = block.strip()
        if not block:
            continue
        if "\n" in block:
            first_line, rest = block.split("\n", 1)
            hint = first_line.strip().lower()
            if hint in LANGUAGE_HINTS:
                block = rest.strip()
            else:
                block = block.strip()
        else:
            if block.strip().lower() in LANGUAGE_HINTS:
                continue
        if block:
            return block

    raise ValueError("No usable code found inside code blocks")


def generate_solution_code(question: str, problem_id: str) -> str:
    prompt = PROMPT_TEMPLATE.format(question=question.strip(), problem_id=problem_id)
    raw_response = call_ollama(prompt)
    # print(f"Raw response: {raw_response}\n")
    return parse_response(raw_response)


def write_ml_file(problem_id: str, code: str) -> Path:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    file_path = GENERATED_DIR / f"{problem_id}.ml"
    file_path.write_text(f"{code.rstrip()}\n", encoding="utf-8")
    return file_path


def run_code(code: str) -> Tuple[bool, str]:
    try:
        proc = subprocess.run(
            ["ocaml", "-stdin"],
            input=code,
            text=True,
            capture_output=True,
        )
    except FileNotFoundError as exc:
        return False, f"ocaml command not found: {exc}"

    output = ""
    if proc.stdout:
        output += proc.stdout
    if proc.stderr:
        output += proc.stderr
    return proc.returncode == 0, output.strip()


def run_via_file(problem_id: str, code: str) -> Tuple[bool, str]:
    file_path = write_ml_file(problem_id, code)
    try:
        proc = subprocess.run(
            ["ocaml", file_path.name],
            cwd=file_path.parent,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        return False, f"ocaml command not found: {exc}"

    output = ""
    if proc.stdout:
        output += proc.stdout
    if proc.stderr:
        output += proc.stderr
    return proc.returncode == 0, output.strip()


def run_subprocess(cmd: List[str], workdir: Path) -> Tuple[bool, str]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=workdir,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        return False, f"{cmd[0]} command not found: {exc}"

    output = ""
    if proc.stdout:
        output += proc.stdout
    if proc.stderr:
        output += proc.stderr
    return proc.returncode == 0, output.strip()


def run_type_check(source_path: Path) -> Tuple[bool, str]:
    return run_subprocess(["ocamlc", "-c", source_path.name], source_path.parent)


def compile_program(source_path: Path, output_name: str) -> Tuple[bool, str]:
    return run_subprocess(["ocamlc", "-o", output_name, source_path.name], source_path.parent)


def run_tests(executable_path: Path) -> Tuple[bool, str]:
    return run_subprocess([f"./{executable_path.name}"], executable_path.parent)


def evaluate_solution(problem_id: str, code: str) -> Dict[str, Tuple[bool, str]]:
    with tempfile.TemporaryDirectory(prefix=f"{problem_id}_") as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        source_path = tmpdir / f"{problem_id}.ml"
        source_path.write_text(f"{code.rstrip()}\n", encoding="utf-8")

        type_result = run_type_check(source_path)
        if type_result[0]:
            compile_result = compile_program(source_path, problem_id)
        else:
            compile_result = (False, "skipped: type check failed")

        if compile_result[0]:
            exec_path = tmpdir / problem_id
            test_result = run_tests(exec_path)
        else:
            test_result = (False, "skipped: compilation failed")

    return {
        "type_check": type_result,
        "compile": compile_result,
        "tests": test_result,
    }


def log_evaluation(problem_id: str, evaluation: Dict[str, Tuple[bool, str]]) -> None:
    def fmt(result: Tuple[bool, str]) -> str:
        return "PASS" if result[0] else "FAIL"

    summary = (
        f"  Type check: {fmt(evaluation['type_check'])}, "
        f"compile: {fmt(evaluation['compile'])}, "
        f"tests: {fmt(evaluation['tests'])}"
    )
    print(summary)

    for label in ["type_check", "compile", "tests"]:
        ok, output = evaluation[label]
        if not ok and output:
            print(f"    [{label} output]")
            for line in output.splitlines():
                print(f"      {line}")


def process_csv(input_file: str, output_file: str):
    problems = []

    with open(input_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        problems.extend(reader)

    results = []
    run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for i, problem in enumerate(problems):
        problem_id = problem["id"]
        question = problem["question"]

        print(f"Processing {problem_id} ({i + 1}/{len(problems)})...")

        solution_error = None
        try:
            solution = generate_solution_code(question, problem_id)
        except Exception as exc:
            print(f"✗ Error processing {problem_id}: {exc}")
            solution = f"ERROR: {exc}"
            run_result = 0
            solution_error = exc
        else:
            try:
                success, output = run_code(solution)
                # if not success:
                #     success, output = run_via_file(problem_id, solution)
            except Exception as exc:
                run_result = 0
                print(f"✗ Failed to run {problem_id}: {exc}")
            else:
                run_result = 1 if success else 0
                status = "✓" if success else "✗"
                print(f"{status} Executed {problem_id} (result={run_result})")
                # if output and not success:
                #     print(f"  Execution output:\n{output}")

        type_check_success = False
        compilation_success = False
        test_success = False
        if solution_error is None:
            evaluation = evaluate_solution(problem_id, solution)
            log_evaluation(problem_id, evaluation)
            type_check_success = evaluation["type_check"][0]
            compilation_success = evaluation["compile"][0]
            test_success = evaluation["tests"][0]

        run_result = 1 if test_success else 0

        results.append(
            {
                "id": problem_id,
                "problem_id": problem_id,
                "question": question,
                "solution": solution,
                "test_cases": "",
                "result": str(run_result),
                "type_check_success": "1" if type_check_success else "0",
                "compilation_success": "1" if compilation_success else "0",
                "test_success": "1" if test_success else "0",
                "timestamp": run_timestamp,
            }
        )

    with open(output_file, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "id",
            "problem_id",
            "question",
            "solution",
            "test_cases",
            "result",
            "type_check_success",
            "compilation_success",
            "test_success",
            "timestamp",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    total = len(results)
    type_passes = sum(1 for row in results if row["type_check_success"] == "1")
    compilation_passes = sum(1 for row in results if row["compilation_success"] == "1")
    test_passes = sum(1 for row in results if row["test_success"] == "1")
    successes = sum(
        1
        for row in results
        if row["type_check_success"] == "1"
        and row["compilation_success"] == "1"
        and row["test_success"] == "1"
    )
    accuracy = (successes / total * 100.0) if total else 0.0
    print(f"\n✓ Results written to {output_file}")
    print(f"Accuracy: {accuracy:.2f}% ({successes}/{total})")

    write_metrics(
        run_timestamp,
        total,
        successes,
        accuracy,
        type_passes,
        compilation_passes,
        test_passes,
    )


def write_metrics(
    timestamp: str,
    total: int,
    successes: int,
    accuracy: float,
    type_passes: int,
    compilation_passes: int,
    test_passes: int,
    path: Path = Path("metrics.csv"),
) -> None:
    fieldnames = [
        "timestamp",
        "total_problems",
        "correct_problems",
        "accuracy",
        "type_check_passed",
        "compilation_passed",
        "test_passed",
    ]
    existing_rows: List[Dict[str, str]] = []
    if path.exists():
        with path.open("r", newline="") as existing:
            reader = csv.DictReader(existing)
            if reader.fieldnames:
                for row in reader:
                    normalized = {field: row.get(field, "") for field in fieldnames}
                    existing_rows.append(normalized)

    existing_rows.append(
        {
            "timestamp": timestamp,
            "total_problems": str(total),
            "correct_problems": str(successes),
            "accuracy": f"{accuracy:.2f}",
            "type_check_passed": str(type_passes),
            "compilation_passed": str(compilation_passes),
            "test_passed": str(test_passes),
        }
    )

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing_rows)


if __name__ == "__main__":
    input_file = "problems.csv"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"output_{timestamp}.csv"
    process_csv(input_file, output_file)
