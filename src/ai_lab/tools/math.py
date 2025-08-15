"""
Math Tool for AI Solutions Lab.

Provides functionality for:
- Evaluating mathematical expressions
- Performing calculations
- Supporting various mathematical operations
- Safe expression evaluation
"""

import asyncio
import math
import re
from typing import Any, Dict, List, Optional, Union


class MathTool:
    """Math tool for evaluating mathematical expressions."""

    def __init__(self):
        """Initialize math tool."""
        # Allowed mathematical functions and constants
        self.allowed_functions = {
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
            "pow": pow,
            "sqrt": math.sqrt,
            "log": math.log,
            "log10": math.log10,
            "exp": math.exp,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "asin": math.asin,
            "acos": math.acos,
            "atan": math.atan,
            "sinh": math.sinh,
            "cosh": math.cosh,
            "tanh": math.tanh,
            "floor": math.floor,
            "ceil": math.ceil,
            "factorial": math.factorial,
            "gcd": math.gcd,
            "lcm": lambda a, b: abs(a * b) // math.gcd(a, b) if a and b else 0,
            "degrees": math.degrees,
            "radians": math.radians,
            "pi": math.pi,
            "e": math.e,
            "inf": float("inf"),
            "nan": float("nan"),
        }

        # Allowed operators
        self.allowed_operators = {"+", "-", "*", "/", "**", "//", "%", "(", ")"}

        # Maximum expression length for safety
        self.max_expression_length = 1000

    async def evaluate(self, input_data: str) -> str:
        """
        Evaluate a mathematical expression.

        Args:
            input_data: Mathematical expression as string

        Returns:
            Result of the calculation as string
        """
        try:
            # Clean and validate input
            expression = self._clean_expression(input_data)

            if not expression:
                return "Error: Empty or invalid expression"

            # Check expression length
            if len(expression) > self.max_expression_length:
                return f"Error: Expression too long (max {self.max_expression_length} characters)"

            # Validate expression safety
            if not self._is_safe_expression(expression):
                return "Error: Expression contains unsafe operations"

            # Evaluate the expression
            result = self._evaluate_expression(expression)

            # Format the result
            return self._format_result(result, expression)

        except Exception as e:
            return f"Math evaluation error: {str(e)}"

    def _clean_expression(self, expression: str) -> str:
        """Clean and normalize the mathematical expression."""
        if not expression:
            return ""

        # Remove extra whitespace
        expression = " ".join(expression.split())

        # Remove common non-mathematical text
        expression = re.sub(r"[^0-9+\-*/().,^%!a-zA-Z\s]", "", expression)

        # Normalize operators
        expression = expression.replace("^", "**")  # Convert ^ to **
        expression = expression.replace("×", "*")  # Convert × to *
        expression = expression.replace("÷", "/")  # Convert ÷ to /

        return expression.strip()

    def _is_safe_expression(self, expression: str) -> bool:
        """Check if the expression is safe to evaluate."""
        # Check for potentially dangerous patterns
        dangerous_patterns = [
            r"__",  # Double underscores (attribute access)
            r"import",  # Import statements
            r"exec",  # Exec function
            r"eval",  # Eval function
            r"open",  # File operations
            r"file",  # File operations
            r"input",  # Input function
            r"raw_input",  # Raw input function
            r"compile",  # Compile function
            r"globals",  # Globals function
            r"locals",  # Locals function
            r"vars",  # Vars function
            r"dir",  # Dir function
            r"getattr",  # Getattr function
            r"setattr",  # Setattr function
            r"delattr",  # Delattr function
            r"hasattr",  # Hasattr function
            r"callable",  # Callable function
            r"isinstance",  # Isinstance function
            r"issubclass",  # Issubclass function
            r"super",  # Super function
            r"property",  # Property function
            r"staticmethod",  # Staticmethod function
            r"classmethod",  # Classmethod function
            r"type",  # Type function
            r"object",  # Object function
            r"Exception",  # Exception class
            r"BaseException",  # BaseException class
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, expression, re.IGNORECASE):
                return False

        # Check for balanced parentheses
        if not self._check_balanced_parentheses(expression):
            return False

        # Check for valid characters
        valid_chars = set(
            "0123456789+-*/().,^%!abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ "
        )
        if not all(c in valid_chars for c in expression):
            return False

        return True

    def _check_balanced_parentheses(self, expression: str) -> bool:
        """Check if parentheses are balanced."""
        stack = []

        for char in expression:
            if char == "(":
                stack.append(char)
            elif char == ")":
                if not stack:
                    return False
                stack.pop()

        return len(stack) == 0

    def _evaluate_expression(self, expression: str) -> Any:
        """Safely evaluate the mathematical expression."""
        # Create a safe namespace with only allowed functions
        safe_namespace = self.allowed_functions.copy()

        # Add basic arithmetic operations
        safe_namespace.update({"True": True, "False": False, "None": None})

        try:
            # Compile the expression
            compiled_expr = compile(expression, "<string>", "eval")

            # Check that the compiled expression only contains allowed names
            for name in compiled_expr.co_names:
                if name not in safe_namespace:
                    raise ValueError(f"Function '{name}' is not allowed")

            # Evaluate the expression
            result = eval(compiled_expr, {"__builtins__": {}}, safe_namespace)

            return result

        except Exception as e:
            raise ValueError(f"Expression evaluation failed: {str(e)}")

    def _format_result(self, result: Any, expression: str) -> str:
        """Format the evaluation result."""
        try:
            if result is None:
                return f"Result of '{expression}': None"

            # Handle different result types
            if isinstance(result, (int, float)):
                # Format numbers
                if isinstance(result, int):
                    formatted_result = str(result)
                else:
                    # Format floats with appropriate precision
                    if result.is_integer():
                        formatted_result = str(int(result))
                    else:
                        formatted_result = (
                            f"{result:.10g}"  # Use g format for smart precision
                        )

                return f"Result of '{expression}': {formatted_result}"

            elif isinstance(result, bool):
                return f"Result of '{expression}': {result}"

            elif isinstance(result, (list, tuple)):
                return f"Result of '{expression}': {result}"

            else:
                return f"Result of '{expression}': {result} (type: {type(result).__name__})"

        except Exception as e:
            return f"Error formatting result: {str(e)}"

    async def calculate_percentage(self, input_data: str) -> str:
        """Calculate percentage values."""
        try:
            # Parse percentage expression (e.g., "15% of 200")
            match = re.match(r"(\d+(?:\.\d+)?)%\s+of\s+(\d+(?:\.\d+)?)", input_data)
            if match:
                percentage = float(match.group(1))
                value = float(match.group(2))
                result = (percentage / 100) * value
                return f"{percentage}% of {value} = {result}"

            # Parse percentage change (e.g., "increase 100 by 15%")
            match = re.match(
                r"increase\s+(\d+(?:\.\d+)?)\s+by\s+(\d+(?:\.\d+)?)%", input_data
            )
            if match:
                base = float(match.group(1))
                percentage = float(match.group(2))
                increase = (percentage / 100) * base
                result = base + increase
                return (
                    f"Increase {base} by {percentage}% = {base} + {increase} = {result}"
                )

            return "Please use format: 'X% of Y' or 'increase X by Y%'"

        except Exception as e:
            return f"Percentage calculation error: {str(e)}"

    async def solve_equation(self, input_data: str) -> str:
        """Solve simple linear equations."""
        try:
            # Parse simple linear equation (e.g., "2x + 3 = 7")
            equation = input_data.replace(" ", "")

            # Check if it's a linear equation
            if "x" not in equation or "=" not in equation:
                return "Please provide a linear equation with 'x' (e.g., '2x + 3 = 7')"

            # Split by equals sign
            parts = equation.split("=")
            if len(parts) != 2:
                return "Invalid equation format. Use 'ax + b = c'"

            left_side = parts[0]
            right_side = parts[1]

            # Simple parsing for ax + b format
            if "x" in left_side:
                # Left side has x
                if "+" in left_side:
                    terms = left_side.split("+")
                    if len(terms) == 2:
                        if "x" in terms[0]:
                            a = (
                                float(terms[0].replace("x", ""))
                                if terms[0] != "x"
                                else 1.0
                            )
                            b = float(terms[1]) if terms[1] else 0.0
                        else:
                            a = (
                                float(terms[1].replace("x", ""))
                                if terms[1] != "x"
                                else 1.0
                            )
                            b = float(terms[0]) if terms[0] else 0.0
                    else:
                        return "Please use format 'ax + b = c'"
                else:
                    a = float(left_side.replace("x", "")) if left_side != "x" else 1.0
                    b = 0.0

                c = float(right_side)

                # Solve: ax + b = c -> x = (c - b) / a
                if a == 0:
                    return "Error: Coefficient of x cannot be zero"

                x = (c - b) / a
                return f"Solution: x = {x}"

            else:
                return "Please put 'x' on the left side of the equation"

        except Exception as e:
            return f"Equation solving error: {str(e)}"

    async def get_math_help(self) -> str:
        """Get help information about supported mathematical operations."""
        help_text = """Math Tool Help

Supported Operations:
- Basic arithmetic: +, -, *, /, ** (power), // (floor division), % (modulo)
- Parentheses: () for grouping
- Functions: abs, round, min, max, sum, pow, sqrt, log, log10, exp
- Trigonometry: sin, cos, tan, asin, acos, atan, sinh, cosh, tanh
- Other: floor, ceil, factorial, gcd, lcm, degrees, radians

Constants: pi, e, inf, nan

Examples:
- Basic: "2 + 3 * 4"
- Functions: "sqrt(16)", "sin(pi/2)"
- Complex: "(2 + 3) * (4 - 1)"
- Percentages: "15% of 200"
- Equations: "2x + 3 = 7"

Safety: Only mathematical operations are allowed. No file access or code execution."""

        return help_text

    async def validate_expression(self, input_data: str) -> str:
        """Validate if an expression is mathematically valid."""
        try:
            expression = self._clean_expression(input_data)

            if not expression:
                return "Error: Empty expression"

            if not self._is_safe_expression(expression):
                return "Error: Expression contains unsafe operations"

            if not self._check_balanced_parentheses(expression):
                return "Error: Unbalanced parentheses"

            # Try to compile the expression
            try:
                compiled_expr = compile(expression, "<string>", "eval")
                return f"Expression '{expression}' is valid and safe to evaluate"
            except SyntaxError as e:
                return f"Syntax error: {str(e)}"
            except Exception as e:
                return f"Compilation error: {str(e)}"

        except Exception as e:
            return f"Validation error: {str(e)}"
