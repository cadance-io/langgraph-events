#!/usr/bin/env python3
"""Validate tests follow BDD structure conventions from CLAUDE.md."""

import ast
import sys
from pathlib import Path


def _parse_file(filepath: Path):
    with open(filepath) as f:
        try:
            return ast.parse(f.read())
        except SyntaxError:
            return None


def _categorize_children(body):
    result = {"describe": [], "when": [], "given": [], "test": []}
    for node in body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            if node.name.startswith("describe_"):
                result["describe"].append(node)
            elif node.name.startswith(("when_", "with_", "without_", "for_")):
                result["when"].append(node)
            elif node.name.startswith("given_"):
                result["given"].append(node)
            elif node.name.startswith(("test_", "it_")):
                result["test"].append(node)
    return result


_CONDITION_INFIXES = ("_when_", "_with_", "_without_")


def _check_embedded_conditions(func, filepath: Path) -> list[str]:
    """Flag test/describe names embedding condition keywords."""
    for infix in _CONDITION_INFIXES:
        if infix in func.name:
            keyword = infix.strip("_")
            return [
                f"{filepath}:{func.lineno} - '{func.name}' embeds "
                f"'{infix}' in its name. Extract the condition "
                f"into a nested {keyword}_* block instead. "
                f"Do not rephrase with conjunctions like 'and'."
            ]
    return []


def _validate_module_level(tree: ast.Module, filepath: Path) -> list[str]:
    """Validate module-level test structure.

    At module level, we expect:
    - describe_* functions (top-level organization)
    - No top-level test_* or it_* functions
    - No top-level when_* functions
    """
    violations = []
    children = _categorize_children(tree.body)

    if children["test"]:
        test_names = [f.name for f in children["test"]]
        violations.append(
            f"{filepath} - Top-level test functions found: {', '.join(test_names)}. "
            f"Tests should be inside describe_* blocks"
        )

    if children["when"]:
        when_names = [f.name for f in children["when"]]
        violations.append(
            f"{filepath} - Top-level when_* functions found: {', '.join(when_names)}. "
            f"when_* blocks should be inside describe_* blocks"
        )

    for describe_func in children["describe"]:
        violations.extend(_validate_describe_block(describe_func, filepath))

    return violations


def _validate_describe_block(describe_func, filepath: Path) -> list[str]:
    """Validate a describe_* block structure.

    In a describe block, we expect:
    - when_* functions for organizing test scenarios
    - given_* functions for precondition grouping (contain when_* blocks)
    - OR test_*/it_* functions for simple tests
    - OR nested describe_* for sub-grouping
    - But NOT a mix of when_* and test_*/it_* at the same level
    """
    violations = []
    children = _categorize_children(describe_func.body)

    if children["when"] and children["test"]:
        violations.append(
            f"{filepath}:{describe_func.lineno} - {describe_func.name} contains both "
            f"when_* and test_*/it_* functions. Use either when_* blocks for complex "
            f"scenarios or test_*/it_* directly for simple tests"
        )

    if not any(children[k] for k in children):
        violations.append(
            f"{filepath}:{describe_func.lineno} - {describe_func.name} is empty "
            f"(no when_*, given_*, test_*/it_*, or nested describe_* functions)"
        )

    violations.extend(_validate_child_funcs(children, filepath))
    return violations


def _validate_given_block(given_func, filepath: Path) -> list[str]:
    """Validate a given_* block structure."""
    violations = []
    children = _categorize_children(given_func.body)

    if children["describe"]:
        nested_names = [f.name for f in children["describe"]]
        violations.append(
            f"{filepath}:{given_func.lineno} - describe_* inside given "
            f"{given_func.name}: {', '.join(nested_names)}. "
            f"describe blocks should be at higher level"
        )

    if not children["test"] and not children["when"]:
        violations.append(
            f"{filepath}:{given_func.lineno} - {given_func.name} has no test_*/it_* "
            f"functions or nested when_* blocks"
        )

    violations.extend(
        _validate_child_funcs(children, filepath, skip={"describe", "given"})
    )
    return violations


def _validate_when_block(when_func, filepath: Path) -> list[str]:
    """Validate a when_* block structure."""
    violations = []
    children = _categorize_children(when_func.body)

    if children["describe"]:
        nested_names = [f.name for f in children["describe"]]
        violations.append(
            f"{filepath}:{when_func.lineno} - describe_* inside when "
            f"{when_func.name}: {', '.join(nested_names)}. "
            f"describe blocks should be at higher level"
        )

    if not children["test"] and not children["when"] and not children["given"]:
        violations.append(
            f"{filepath}:{when_func.lineno} - {when_func.name} has no test_*/it_* "
            f"functions or nested when_*/given_* blocks"
        )

    violations.extend(_validate_child_funcs(children, filepath, skip={"describe"}))
    return violations


def _validate_child_funcs(
    children,
    filepath,
    *,
    skip: set[str] | frozenset[str] = frozenset(),
):
    violations = []
    for func in children.get("when", []):
        if "when" not in skip:
            violations.extend(_validate_when_block(func, filepath))
    for func in children.get("given", []):
        if "given" not in skip:
            violations.extend(_validate_given_block(func, filepath))
    for func in children.get("describe", []):
        if "describe" not in skip:
            violations.extend(_validate_describe_block(func, filepath))
    for func in children.get("test", []):
        if "test" not in skip:
            violations.extend(_validate_test_function(func, filepath))
    return violations


def _validate_embedded_conditions_everywhere(
    tree: ast.AST,
    filepath: Path,
) -> list[str]:
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            violations.extend(_check_embedded_conditions(node, filepath))
    return violations


def _validate_test_function(test_func, filepath: Path) -> list[str]:
    """Validate a test_*/it_* function is a leaf node."""
    violations = []
    for node in test_func.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            if node.name.startswith(("test_", "it_", "when_", "describe_")):
                violations.append(
                    f"{filepath}:{test_func.lineno} - Test function {test_func.name} "
                    f"contains nested function {node.name}. Test functions should be "
                    f"leaf nodes"
                )
    return violations


def validate_directory(path: Path) -> int:
    """Validate all test files in directory or a single test file."""
    all_violations = []

    if path.is_file():
        test_files = (
            [path]
            if path.name.startswith("test_") or path.name == "conftest.py"
            else []
        )
    else:
        test_files = list(path.rglob("test_*.py")) + list(path.rglob("conftest.py"))

    for test_file in test_files:
        if "scripts" in test_file.parts or test_file.name.startswith("_"):
            continue

        tree = _parse_file(test_file)
        if tree is not None:
            all_violations.extend(
                _validate_embedded_conditions_everywhere(tree, test_file)
            )
            all_violations.extend(_validate_module_level(tree, test_file))

    if all_violations:
        print("❌ Test validation failed:\n", file=sys.stderr)
        for violation in all_violations:
            print(f"  {violation}", file=sys.stderr)
        print(f"\nTotal violations: {len(all_violations)}", file=sys.stderr)
        return 1

    print(f"✅ All {len(test_files)} test files pass validation", file=sys.stderr)
    return 0


def main():
    """Main entry point."""
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tests")

    if not path.exists():
        print(f"Error: Path {path} does not exist")
        sys.exit(1)

    sys.exit(validate_directory(path))


if __name__ == "__main__":
    main()
