"""Constants for evaluation metrics and thresholds."""

# Reward thresholds for pass/fail determination
PASS_THRESHOLD = 1.0
TYPE_CHECK_THRESHOLD = 0.25
COMPILE_THRESHOLD = 0.10
TEST_THRESHOLD = 0.65

# Pattern to failure stage mapping (checked in order)
FAILURE_STAGE_PATTERNS = [
    ("syntax error", "type_check:syntax"),
    ("type error", "type_check:type"),
    ("timeout (type_check)", "type_check:timeout"),
    ("compile failure", "compile"),
    ("timeout (compile)", "compile:timeout"),
    ("unbound module", "compile:unbound_module"),
    ("unbound value", "compile:unbound_value"),
    ("test failure", "execution:test_fail"),
    ("timeout (tests)", "execution:timeout"),
    ("repetitive", "degenerate:repetitive"),
    ("low code ratio", "degenerate:low_code_ratio"),
    ("code purity", "degenerate:low_code_ratio"),
    ("code block spam", "degenerate:code_block_spam"),
    ("stub", "degenerate:stub"),
    ("empty", "degenerate:empty"),
    ("too short", "degenerate:empty"),
]
