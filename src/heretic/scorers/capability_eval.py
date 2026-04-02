# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import json
import re
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from heretic.plugin import Context
from heretic.scorer import Score, Scorer
from heretic.utils import Prompt, print

# Valid A::B tokens.
_AB_TOKENS = frozenset({"A#", "#A", "B#", "#B"})

_AB_QUESTION_TEMPLATE = """A::B is a system with 4 tokens: `A#`, `#A`, `B#` and `#B`.

An A::B program is a sequence of tokens. Example:

    B# A# #B #A B#

To *compute* a program, we must rewrite neighbor tokens, using the rules:

    A# #A ... becomes ... nothing
    A# #B ... becomes ... #B A#
    B# #A ... becomes ... #A B#
    B# #B ... becomes ... nothing

In other words, whenever two neighbor tokens have their '#' facing each-other,
they must be rewritten according to the corresponding rule.

Now, consider the following program:

{program}

Return the final state of the program.
"""

# Hand-verified A::B tasks: (initial_program, expected_final_state).
_AB_TASKS: list[tuple[str, str]] = [
    ("#B #B B# #B A# A# A# #A B# #B A# #A B# #A #A", "#B #B B#"),
    ("A# B# A# #A #B A# B# #A B# #A B# #B A# #B A#", "B# A# A#"),
    ("B# #A #A #A #B #A #A A# B# #A #B B# #A A# #B", "#A #A #A #A #A #A A#"),
    ("#A #B #B A# #A #B #A B# #A #A #A B# #B #B #B", "#A #B #B #B #A #A #A #A #B"),
    ("B# #A #B #A B# #A #B A# #B #B B# #B #B #A A#", "#A #A #A #B #B #B A#"),
    ("A# #B B# #B #B #B #A #B #B A# #B B# #A B# B#", "#B #B #B #B #B #B B# B# B#"),
    ("B# #B #B B# #B #B A# B# #A #A #B A# B# A# A#", "#B #B #A A# B# A# A#"),
    ("B# B# #A #B A# #B B# #B #A A# #A A# #B #A A#", "#A #B A#"),
    ("A# #B #A #A A# B# #A A# B# A# B# A# #A B# #A", "#B #A B# A# B# B# B#"),
    ("#A B# #B #A A# B# #B #A #A #B #A #A #A A# #B", "#A #A #A #B #A #A #A #B A#"),
    ("B# B# #A #A #A #A B# #B A# B# A# B# #B A# #B", "#A #A #A #A B# B# A# A# A#"),
    ("A# #B A# B# #A B# A# A# #A B# B# #A A# B# #B", "#B A# B# B# B# B# A#"),
    ("#B #A B# #A #B A# #B A# #A A# #A A# A# #B A#", "#B #A #A #B #B A# A# A# A#"),
    ("#A A# #A A# A# #A #B A# #A A# #B A# #B B# A#", "#A #B #B #B A# A# A# B# A#"),
    (
        "#B #A B# B# B# B# A# A# B# B# A# #B #B A# B#",
        "#B #A B# B# B# B# A# A# A# A# B#",
    ),
    ("B# #A A# B# A# #B B# B# A# B# A# #B #B #A #B", "#A B# A# A# A#"),
]


def _extract_ab_answer(response: str) -> str:
    """
    Extract the A::B token sequence from a model response.

    Scans lines from the end upward, returning the first line that consists
    entirely of valid A::B tokens. Falls back to the stripped response.
    """
    cleaned = response.replace("`", "").strip()
    for line in reversed(cleaned.splitlines()):
        tokens = line.split()
        if tokens and all(token in _AB_TOKENS for token in tokens):
            return " ".join(tokens)
    return cleaned


def _score_ab(response: str, expected: str, metadata: dict[str, Any]) -> float:
    """Score an A::B challenge response. Awards 1.0 for a correct answer."""
    return 1.0 if _extract_ab_answer(response) == expected else 0.0


def _build_ab_tasks() -> list[tuple[str, str, dict[str, Any]]]:
    """Build A::B challenge tasks as (question, answer, metadata) triples."""
    tasks: list[tuple[str, str, dict[str, Any]]] = []
    for index, (program, answer) in enumerate(_AB_TASKS):
        question = _AB_QUESTION_TEMPLATE.format(program=program)
        metadata = {"source_dataset": "ab", "source_index": index}
        tasks.append((question, answer, metadata))
    return tasks


# ──────────────────────────────────────────────────
# Advanced geometry challenge
# ──────────────────────────────────────────────────

_GEOMETRY_FORMAT_INSTRUCTIONS = (
    "For all geometry problems:\n"
    "1. Give coordinates in the form (x, y)\n"
    "2. Round decimal answers to 3 decimal places\n"
    "3. Use the degree symbol \u00b0 for angles\n"
    "4. Return only the angle, coordinates, or radius as your answer.\n"
)

# Hand-verified geometry tasks: (task_description, expected_answer, task_type).
_ADVANCED_GEOMETRY_TASKS: list[tuple[str, str, str]] = [
    (
        "Given a triangle with vertices A=(10, -6), B=(10, 9), and C=(-2, -7), determine the angle at B in degrees.",
        "36.870\u00b0",
        "angle_measure",
    ),
    (
        "Consider triangle ABC with coordinates A=(-6, 7), B=(6, -1), and C=(-4, 7). Compute the radius of its incircle.",
        "0.547",
        "incircle_radius",
    ),
    (
        "Given a triangle with vertices A=(-2, 8), B=(-8, 3), and C=(6, 0), determine the angle at B in degrees.",
        "51.900\u00b0",
        "angle_measure",
    ),
    (
        "Consider triangle ABC with coordinates A=(-7, 0), B=(-10, 4), and C=(-6, 7). Compute the radius of its incircle.",
        "1.464",
        "incircle_radius",
    ),
    (
        "In triangle ABC with coordinates A=(2, 3), B=(-1, 7), and C=(1, 2), find the measure (in degrees) of angle ABC.",
        "15.068\u00b0",
        "angle_measure",
    ),
    (
        "Given a triangle with vertices A=(-3, 0), B=(-4, 9), and C=(5, -2), determine the angle at B in degrees.",
        "32.949\u00b0",
        "angle_measure",
    ),
    (
        "For triangle with vertices A=(2, -1), B=(8, -5), and C=(-6, -7), determine the orthocenter (intersection of altitudes).",
        "(1.294, 3.941)",
        "orthocenter",
    ),
    (
        "For triangle with vertices A=(9, 6), B=(8, -6), and C=(-2, 3), determine the orthocenter (intersection of altitudes).",
        "(5.721, 2.357)",
        "orthocenter",
    ),
    (
        "Given triangle ABC with coordinates A=(-1, -1), B=(8, 2), and C=(-9, 2), find the coordinates of its orthocenter.",
        "(-1.000, -22.000)",
        "orthocenter",
    ),
    (
        "Given triangle ABC with coordinates A=(-5, 8), B=(5, -5), and C=(9, -8), find the coordinates of its orthocenter.",
        "(-52.455, -55.273)",
        "orthocenter",
    ),
    (
        "Find the incircle radius of triangle ABC whose vertices are A=(5, -5), B=(-1, 6), and C=(-5, 8).",
        "0.958",
        "incircle_radius",
    ),
    (
        "Given triangle ABC with coordinates A=(5, 4), B=(-10, 2), and C=(9, -8), find the coordinates of its orthocenter.",
        "(6.915, 7.638)",
        "orthocenter",
    ),
    (
        "For triangle with vertices A=(5, -6), B=(4, 4), and C=(-4, -2), determine the orthocenter (intersection of altitudes).",
        "(1.581, -1.442)",
        "orthocenter",
    ),
    (
        "For triangle with vertices A=(4, -6), B=(-8, -3), and C=(-2, 3), determine the orthocenter (intersection of altitudes).",
        "(-2.600, 0.600)",
        "orthocenter",
    ),
    (
        "Find the incircle radius of triangle ABC whose vertices are A=(-2, 10), B=(-4, -5), and C=(-1, 7).",
        "0.685",
        "incircle_radius",
    ),
    (
        "For triangle with vertices A=(0, -9), B=(1, 0), and C=(6, -10), determine the orthocenter (intersection of altitudes).",
        "(-0.545, -9.273)",
        "orthocenter",
    ),
]

# Pattern matching a number next to a degree symbol or the word "degrees".
_ANGLE_PATTERN = re.compile(r"(-?\d+\.?\d*)\s*(?:\u00b0|degrees?)")

# Pattern matching a coordinate pair like (x, y).
_COORDINATE_PATTERN = re.compile(r"\(\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)")

# Pattern matching a decimal number (must contain a decimal point).
_DECIMAL_PATTERN = re.compile(r"-?\d+\.\d+")


def _extract_angle(response: str) -> float | None:
    """Extract an angle value from a model response."""
    matches = _ANGLE_PATTERN.findall(response)
    if matches:
        return float(matches[-1])
    # Fall back to the last decimal number.
    decimals = _DECIMAL_PATTERN.findall(response)
    if decimals:
        return float(decimals[-1])
    return None


def _extract_coordinates(response: str) -> tuple[float, float] | None:
    """Extract a coordinate pair from a model response."""
    matches = _COORDINATE_PATTERN.findall(response)
    if matches:
        x, y = matches[-1]
        return float(x), float(y)
    return None


def _extract_decimal(response: str) -> float | None:
    """Extract a decimal number from a model response."""
    matches = _DECIMAL_PATTERN.findall(response)
    if matches:
        return float(matches[-1])
    return None


def _score_advanced_geometry(
    response: str, expected: str, metadata: dict[str, Any]
) -> float:
    """Score an advanced geometry response with task-specific partial credit."""
    task_type = metadata["task_type"]
    cleaned = response.replace("`", "").strip()

    try:
        if task_type == "angle_measure":
            expected_value = float(expected.replace("\u00b0", ""))
            extracted = _extract_angle(cleaned)
            if extracted is not None and f"{extracted:.3f}" == f"{expected_value:.3f}":
                return 1.0
            return 0.05 if cleaned else 0.01

        if task_type == "orthocenter":
            expected_coords = _extract_coordinates(expected)
            extracted = _extract_coordinates(cleaned)
            if expected_coords is not None and extracted is not None:
                if (
                    f"{extracted[0]:.3f}" == f"{expected_coords[0]:.3f}"
                    and f"{extracted[1]:.3f}" == f"{expected_coords[1]:.3f}"
                ):
                    return 1.0
            return 0.01

        if task_type == "incircle_radius":
            expected_value = float(expected)
            extracted = _extract_decimal(cleaned)
            if extracted is not None:
                if f"{extracted:.3f}" == f"{expected_value:.3f}":
                    return 1.0
                if f"{extracted:.2f}" == f"{expected_value:.2f}":
                    return 0.5
                return 0.05
            return 0.01
    except (ValueError, IndexError):
        pass

    return 0.01


def _build_advanced_geometry_tasks() -> list[tuple[str, str, dict[str, Any]]]:
    """Build advanced geometry tasks as (question, answer, metadata) triples."""
    tasks: list[tuple[str, str, dict[str, Any]]] = []
    for index, (description, answer, task_type) in enumerate(_ADVANCED_GEOMETRY_TASKS):
        question = f"{description} {_GEOMETRY_FORMAT_INSTRUCTIONS}"
        metadata = {
            "source_dataset": "advanced_geometry",
            "source_index": index,
            "task_type": task_type,
        }
        tasks.append((question, answer, metadata))
    return tasks


# ──────────────────────────────────────────────────
# Knights and knaves challenge
# ──────────────────────────────────────────────────

# All valid role words across all terminology variants.
_VALID_ROLES = frozenset(
    {
        "knight",
        "knave",
        "pioneer",
        "laggard",
        "saint",
        "sinner",
        "hero",
        "villain",
        "angel",
        "devil",
        "altruist",
        "egoist",
        "sage",
        "fool",
    }
)

# Hand-verified knights and knaves tasks: (question, expected_answer).
_KNIGHTS_KNAVES_TASKS: list[tuple[str, str]] = [
    (
        'A very special island is inhabited only by altruists and egoists. Altruists always tell the truth, and egoists always lie. You meet 4 inhabitants: Samuel, Chloe, Owen, and Mason. Samuel remarked, "Owen is an altruist". "(it is not the case that (Samuel is an egoist or Chloe is an altruist)) or (if (if Samuel is an egoist then Owen is an egoist) then Mason is an altruist) or ((Owen is an altruist) and (Mason is an altruist and Samuel is an egoist and Samuel is an altruist and Mason is an egoist) and (Mason is an altruist if and only if Chloe is an altruist) and (Samuel is an egoist and Owen is an egoist and Chloe is an altruist and Mason is an egoist)) or (it is not the case that (Samuel is an egoist))," Chloe claimed. Owen asserted: "(Mason is an egoist and (Samuel is an egoist if and only if Owen is an altruist)) and (if (Samuel is an altruist) then (Mason is an egoist and Mason is an altruist))". Mason stated, "Samuel is an altruist". So who is an altruist and who is an egoist? (Format your answer like: "Samuel is a altruist/egoist, Chloe is a altruist/egoist, Owen is a altruist/egoist, and Mason is a altruist/egoist")',
        "Samuel is an egoist, Chloe is an egoist, Owen is an egoist, and Mason is an egoist.",
    ),
    (
        'A very special island is inhabited only by knights and knaves. Knights always tell the truth, and knaves always lie. You meet 4 inhabitants: James, Samuel, Noah, and Evelyn. "it is not the case that ((Evelyn is a knave or Samuel is a knave or Noah is a knave or Evelyn is a knight) or (Evelyn is a knight or James is a knight or Evelyn is a knave) or (Noah is a knight if and only if Noah is a knave))," James declared. "((Samuel is a knave) and (Evelyn is a knight if and only if Noah is a knight) and (James is a knave or Noah is a knight)) or (if (James is a knave or Evelyn is a knave or Noah is a knight or Samuel is a knight) then (if James is a knight then Samuel is a knight)) or ((Samuel is a knave) and (if James is a knave then Evelyn is a knight))," Samuel declared. As Noah put it, "Samuel is a knave". "Noah is a knave if and only if (if (if James is a knave then Noah is a knave) then (if Evelyn is a knight then James is a knave))," Evelyn mentioned. So who is a knight and who is a knave? (Format your answer like: "James is a knight/knave, Samuel is a knight/knave, Noah is a knight/knave, and Evelyn is a knight/knave")',
        "James is a knave, Samuel is a knight, Noah is a knave, and Evelyn is a knight.",
    ),
    (
        'A very special island is inhabited only by heros and villains. Heros always tell the truth, and villains always lie. You meet 4 inhabitants: William, Oliver, Ella, and Liam. William said that ((if Liam is a hero then Ella is a villain) if and only if (Ella is a hero or Oliver is a hero or Ella is a villain or William is a hero)) or ((Ella is a villain or Oliver is a hero or William is a hero or Oliver is a villain) if and only if Oliver is a hero). Oliver remarked, "((William is a hero and William is a villain) if and only if (if Liam is a villain then Oliver is a hero)) or Liam is a villain or (if (Liam is a villain) then (William is a villain if and only if Ella is a villain))". Ella told you that it is not the case that ((William is a hero if and only if William is a villain) and (Liam is a hero and Oliver is a villain) and (William is a villain or Oliver is a villain)). According to Liam, "Ella is a villain". So who is a hero and who is a villain? (Format your answer like: "William is a hero/villain, Oliver is a hero/villain, Ella is a hero/villain, and Liam is a hero/villain")',
        "William is a hero, Oliver is a hero, Ella is a hero, and Liam is a villain.",
    ),
    (
        'A very special island is inhabited only by knights and knaves. Knights always tell the truth, and knaves always lie. You meet 4 inhabitants: Avery, Noah, Emily, and William. Avery commented, "if (if (Noah is a knave if and only if William is a knave) then (William is a knight and Noah is a knight and Emily is a knight)) then ((if William is a knight then Emily is a knave) or (Emily is a knight))". Noah commented, "((William is a knight and William is a knave and Avery is a knave) and (William is a knave) and (Emily is a knave and William is a knight and Avery is a knight)) if and only if ((Noah is a knight and Avery is a knight and William is a knight) and (William is a knight and Avery is a knave) and (if Emily is a knave then Noah is a knight))". As Emily put it, "((Avery is a knave and William is a knave and Noah is a knave) if and only if (if Noah is a knave then Avery is a knave)) or ((Avery is a knight) if and only if (William is a knight if and only if Noah is a knave)) or (William is a knight if and only if (Noah is a knave if and only if Emily is a knight)) or (if (William is a knave or William is a knight or Avery is a knave) then (William is a knight or Emily is a knight))". According to William, "(it is not the case that (if Noah is a knight then Noah is a knave)) and ((Noah is a knight and Noah is a knave) or (Noah is a knight and Avery is a knave and Avery is a knight)) and (if (Avery is a knight and William is a knight) then Avery is a knave) and ((William is a knight or Emily is a knight or Avery is a knight or Noah is a knight) or (Emily is a knight if and only if Avery is a knave) or William is a knight or (if Avery is a knight then Noah is a knave))". So who is a knight and who is a knave? (Format your answer like: "Avery is a knight/knave, Noah is a knight/knave, Emily is a knight/knave, and William is a knight/knave")',
        "Avery is a knight, Noah is a knight, Emily is a knight, and William is a knave.",
    ),
    (
        'A very special island is inhabited only by sages and fools. Sages always tell the truth, and fools always lie. You meet 4 inhabitants: Aria, Liam, Ella, and Riley. Aria remarked, "Liam is a sage or ((Liam is a fool if and only if Riley is a sage) and (Ella is a sage and Aria is a sage and Liam is a fool and Liam is a sage))". In a statement by Liam: "it is not the case that ((if Aria is a fool then Aria is a sage) if and only if (if Ella is a sage then Aria is a sage))". "((if Ella is a sage then Riley is a sage) or (Riley is a fool if and only if Liam is a sage) or (Aria is a sage and Ella is a sage and Riley is a fool) or Liam is a fool) or (if (Ella is a sage or Aria is a sage or Aria is a fool) then (Riley is a sage and Ella is a sage and Aria is a sage and Aria is a fool)) or ((Ella is a sage if and only if Aria is a sage) and (if Riley is a sage then Riley is a fool) and (Aria is a sage or Liam is a sage or Riley is a sage) and (Liam is a fool or Aria is a fool or Liam is a sage))," Ella claimed. "Ella is a fool" - Riley. So who is a sage and who is a fool? (Format your answer like: "Aria is a sage/fool, Liam is a sage/fool, Ella is a sage/fool, and Riley is a sage/fool")',
        "Aria is a fool, Liam is a fool, Ella is a sage, and Riley is a fool.",
    ),
    (
        'A very special island is inhabited only by knights and knaves. Knights always tell the truth, and knaves always lie. You meet 4 inhabitants: Daniel, Michael, Jacob, and Evelyn. "Michael is a knave," Daniel claimed. Michael expressed that ((Daniel is a knave if and only if Evelyn is a knave) or Evelyn is a knave or (if Michael is a knight then Evelyn is a knave) or (if Jacob is a knave then Jacob is a knight)) if and only if ((Daniel is a knave and Jacob is a knave and Michael is a knight) if and only if (if Michael is a knight then Jacob is a knight)). Jacob was heard saying, "Evelyn is a knave or ((Daniel is a knave or Michael is a knight) and (Daniel is a knight and Evelyn is a knave and Daniel is a knave) and Michael is a knave) or ((Evelyn is a knave or Daniel is a knave) and Evelyn is a knave)". Evelyn commented, "Michael is a knight and ((Jacob is a knight) and (if Daniel is a knave then Michael is a knight) and (Michael is a knave and Michael is a knight) and (Michael is a knave and Daniel is a knave))". So who is a knight and who is a knave? (Format your answer like: "Daniel is a knight/knave, Michael is a knight/knave, Jacob is a knight/knave, and Evelyn is a knight/knave")',
        "Daniel is a knight, Michael is a knave, Jacob is a knight, and Evelyn is a knave.",
    ),
    (
        'A very special island is inhabited only by knights and knaves. Knights always tell the truth, and knaves always lie. You meet 4 inhabitants: Sofia, David, Victoria, and Abigail. According to Sofia, "(if (David is a knave if and only if Abigail is a knight) then (David is a knight and David is a knave and Victoria is a knight)) and ((Victoria is a knave and David is a knight and Victoria is a knight and Abigail is a knave) if and only if (Sofia is a knight if and only if Victoria is a knight)) and ((Victoria is a knight and Victoria is a knave and David is a knight) or (Victoria is a knave if and only if David is a knight) or (David is a knave if and only if David is a knight)) and ((David is a knave and Victoria is a knave and Sofia is a knight) if and only if (Abigail is a knave if and only if Victoria is a knight))". David told you that if (if Victoria is a knave then (David is a knave)) then ((Victoria is a knave if and only if Abigail is a knight) or (David is a knight or Victoria is a knave) or (Victoria is a knave)). According to Victoria, "David is a knight". "((David is a knave) if and only if (Sofia is a knave and Victoria is a knight and Victoria is a knave and David is a knight)) if and only if (Abigail is a knight or (Victoria is a knight or Victoria is a knave or David is a knight) or (if David is a knight then Victoria is a knight) or (Abigail is a knight and Sofia is a knight and Victoria is a knave and David is a knight))" - Abigail. So who is a knight and who is a knave? (Format your answer like: "Sofia is a knight/knave, David is a knight/knave, Victoria is a knight/knave, and Abigail is a knight/knave")',
        "Sofia is a knave, David is a knight, Victoria is a knight, and Abigail is a knight.",
    ),
    (
        'A very special island is inhabited only by knights and knaves. Knights always tell the truth, and knaves always lie. You meet 4 inhabitants: Oliver, James, Avery, and Owen. As Oliver put it, "Owen is a knave". In James\'s words: "if ((Owen is a knight if and only if Owen is a knave) or (Avery is a knave)) then ((Owen is a knave or James is a knight or Avery is a knight or Oliver is a knight) or (if Avery is a knight then Oliver is a knave) or (James is a knight if and only if Owen is a knight))". Avery was heard saying, "(it is not the case that (James is a knave)) or ((Oliver is a knight if and only if Owen is a knave) or (if Owen is a knight then Oliver is a knight) or (Avery is a knight and James is a knight and Owen is a knave and Oliver is a knight))". In Owen\'s words: "((Avery is a knight if and only if James is a knave) and (Owen is a knave) and (if James is a knight then James is a knave)) and ((Avery is a knight) or Avery is a knave or (Owen is a knight or James is a knight or Avery is a knave)) and ((James is a knave if and only if Oliver is a knave) if and only if (Avery is a knight)) and (if Oliver is a knave then (James is a knave))". So who is a knight and who is a knave? (Format your answer like: "Oliver is a knight/knave, James is a knight/knave, Avery is a knight/knave, and Owen is a knight/knave")',
        "Oliver is a knight, James is a knight, Avery is a knight, and Owen is a knave.",
    ),
    (
        'A very special island is inhabited only by knights and knaves. Knights always tell the truth, and knaves always lie. You meet 4 inhabitants: Logan, Owen, Jacob, and Michael. Logan remarked, "((Logan is a knight and Jacob is a knight) and (Owen is a knave if and only if Logan is a knight) and Jacob is a knave) if and only if ((Owen is a knight or Michael is a knave or Owen is a knave or Michael is a knight) or (if Logan is a knight then Michael is a knight))". "Logan is a knave," Owen mentioned. Jacob said that if (it is not the case that (Logan is a knave or Owen is a knave or Jacob is a knight)) then (Logan is a knave and (Michael is a knave) and (Owen is a knight and Owen is a knave and Michael is a knight and Jacob is a knight)). As Michael put it, "it is not the case that ((Jacob is a knight or Owen is a knight or Logan is a knight or Owen is a knave) or (if Logan is a knight then Logan is a knave) or Owen is a knight or Jacob is a knight)". So who is a knight and who is a knave? (Format your answer like: "Logan is a knight/knave, Owen is a knight/knave, Jacob is a knight/knave, and Michael is a knight/knave")',
        "Logan is a knave, Owen is a knight, Jacob is a knight, and Michael is a knave.",
    ),
    (
        'A very special island is inhabited only by sages and fools. Sages always tell the truth, and fools always lie. You meet 4 inhabitants: Jacob, Amelia, Penelope, and Mason. In a statement by Jacob: "((Mason is a fool or Amelia is a sage) and (Penelope is a sage if and only if Amelia is a sage) and (Amelia is a sage or Jacob is a sage or Penelope is a sage or Mason is a sage) and Penelope is a sage) if and only if ((Jacob is a sage or Penelope is a fool) and Amelia is a fool and Amelia is a sage)". According to Amelia, "Jacob is a sage and Mason is a fool". Penelope asserted: "if ((if Amelia is a fool then Jacob is a fool) or (Jacob is a fool) or (Amelia is a sage and Jacob is a sage and Penelope is a sage)) then (Jacob is a sage and (Amelia is a fool if and only if Mason is a sage) and (Penelope is a sage and Jacob is a sage and Mason is a sage and Jacob is a fool) and Amelia is a fool)". Mason expressed that if ((Jacob is a fool) if and only if (if Penelope is a fool then Penelope is a sage)) then ((Mason is a fool) or (Penelope is a sage or Jacob is a sage or Amelia is a fool or Amelia is a sage) or (if Penelope is a fool then Amelia is a sage) or (Jacob is a sage and Mason is a sage)). So who is a sage and who is a fool? (Format your answer like: "Jacob is a sage/fool, Amelia is a sage/fool, Penelope is a sage/fool, and Mason is a sage/fool")',
        "Jacob is a sage, Amelia is a fool, Penelope is a sage, and Mason is a sage.",
    ),
    (
        'A very special island is inhabited only by sages and fools. Sages always tell the truth, and fools always lie. You meet 4 inhabitants: Riley, Luke, Henry, and Liam. Riley commented, "((Luke is a sage and Liam is a fool) and Riley is a sage and (Liam is a fool)) or (if (Henry is a fool if and only if Luke is a fool) then (Luke is a fool if and only if Riley is a sage)) or ((Liam is a sage or Luke is a fool or Henry is a fool) if and only if (Luke is a sage and Luke is a fool and Riley is a sage)) or (it is not the case that (if Luke is a fool then Henry is a sage))". Luke expressed that Riley is a sage or ((Henry is a fool or Luke is a sage or Henry is a sage or Liam is a sage) and (Liam is a fool)) or ((Riley is a sage or Luke is a sage or Liam is a fool or Henry is a sage) and (Riley is a fool if and only if Riley is a sage) and (Liam is a fool and Henry is a fool and Henry is a sage and Riley is a sage) and (Luke is a sage if and only if Liam is a fool)) or (if (Riley is a sage) then Liam is a sage). According to Henry, "if ((Riley is a fool and Henry is a sage and Liam is a sage and Riley is a sage) if and only if (Liam is a fool)) then (if (Henry is a sage or Luke is a sage) then (if Luke is a fool then Luke is a sage))". In a statement by Liam: "(Riley is a fool and (if Riley is a fool then Henry is a fool) and (Liam is a sage or Henry is a sage or Riley is a sage)) and Henry is a sage". So who is a sage and who is a fool? (Format your answer like: "Riley is a sage/fool, Luke is a sage/fool, Henry is a sage/fool, and Liam is a sage/fool")',
        "Riley is a sage, Luke is a sage, Henry is a sage, and Liam is a fool.",
    ),
    (
        'A very special island is inhabited only by angels and devils. Angels always tell the truth, and devils always lie. You meet 4 inhabitants: Luke, Daniel, Chloe, and Isabella. According to Luke, "Isabella is an angel". In a statement by Daniel: "it is not the case that ((Chloe is a devil) and (Chloe is a devil or Chloe is an angel or Luke is an angel))". Chloe said that (Luke is an angel if and only if (Daniel is a devil and Isabella is an angel)) and ((Luke is a devil or Daniel is a devil or Daniel is an angel) if and only if Isabella is a devil) and (it is not the case that (Isabella is an angel)) and (it is not the case that (Luke is an angel and Luke is a devil and Chloe is an angel)). Isabella told you that if (Luke is an angel and (Daniel is an angel or Chloe is an angel) and (Daniel is a devil or Luke is an angel)) then ((Isabella is an angel and Daniel is an angel and Chloe is an angel) and (Isabella is an angel or Chloe is a devil or Luke is a devil)). So who is an angel and who is a devil? (Format your answer like: "Luke is a angel/devil, Daniel is a angel/devil, Chloe is a angel/devil, and Isabella is a angel/devil")',
        "Luke is an angel, Daniel is a devil, Chloe is a devil, and Isabella is an angel.",
    ),
    (
        'A very special island is inhabited only by heros and villains. Heros always tell the truth, and villains always lie. You meet 4 inhabitants: Olivia, Noah, William, and Aiden. In a statement by Olivia: "((Noah is a villain) or (if Noah is a hero then William is a villain) or (Noah is a hero) or (Noah is a villain and Aiden is a hero)) and (it is not the case that (if William is a hero then Noah is a hero)) and Noah is a villain". "((Aiden is a villain if and only if Olivia is a villain) if and only if (Olivia is a hero)) and ((Aiden is a villain if and only if Aiden is a hero) and Olivia is a villain and (Noah is a hero if and only if William is a hero))" - Noah. William commented, "Aiden is a hero". Aiden said, "if ((William is a villain and William is a hero and Olivia is a villain) or (Aiden is a hero or Olivia is a villain or Noah is a villain or William is a villain) or (if William is a hero then William is a villain)) then (William is a villain if and only if (William is a hero))." So who is a hero and who is a villain? (Format your answer like: "Olivia is a hero/villain, Noah is a hero/villain, William is a hero/villain, and Aiden is a hero/villain")',
        "Olivia is a villain, Noah is a villain, William is a villain, and Aiden is a villain.",
    ),
    (
        'A very special island is inhabited only by altruists and egoists. Altruists always tell the truth, and egoists always lie. You meet 4 inhabitants: Avery, Luke, Harper, and Benjamin. Avery noted, "((if Benjamin is an egoist then Harper is an altruist) if and only if (Benjamin is an altruist or Avery is an altruist)) if and only if Luke is an egoist". Luke stated, "if ((Avery is an altruist if and only if Harper is an egoist) or Avery is an altruist) then ((if Avery is an egoist then Benjamin is an egoist) and (Avery is an altruist if and only if Benjamin is an altruist) and (if Luke is an altruist then Avery is an egoist) and (if Benjamin is an altruist then Avery is an egoist))". Harper commented, "if ((Luke is an egoist and Avery is an egoist and Luke is an altruist) if and only if (Benjamin is an altruist)) then (if Luke is an altruist then (Luke is an egoist))". Benjamin asserted: "(Luke is an egoist or Harper is an egoist or Avery is an egoist) or (Benjamin is an altruist if and only if (Harper is an altruist))". So who is an altruist and who is an egoist? (Format your answer like: "Avery is a altruist/egoist, Luke is a altruist/egoist, Harper is a altruist/egoist, and Benjamin is a altruist/egoist")',
        "Avery is an altruist, Luke is an egoist, Harper is an altruist, and Benjamin is an altruist.",
    ),
    (
        'A very special island is inhabited only by angels and devils. Angels always tell the truth, and devils always lie. You meet 4 inhabitants: Ava, Charlotte, Jack, and Chloe. In a statement by Ava: "(Chloe is an angel) if and only if (if Charlotte is an angel then (Charlotte is a devil if and only if Jack is an angel))". Charlotte was heard saying, "(Jack is an angel if and only if (if Ava is a devil then Jack is an angel)) or Ava is an angel or (if (Jack is an angel or Jack is a devil or Ava is an angel) then (Chloe is a devil and Jack is an angel and Jack is a devil)) or Chloe is an angel". "(if (Chloe is an angel or Charlotte is a devil) then (if Chloe is an angel then Charlotte is an angel)) and (Charlotte is a devil if and only if (Jack is a devil)) and (if (Ava is an angel) then (Charlotte is a devil if and only if Chloe is a devil)) and Ava is an angel," Jack mentioned. Chloe said, "((Charlotte is a devil if and only if Charlotte is an angel) and (Charlotte is an angel or Ava is a devil or Ava is an angel or Chloe is an angel) and (Ava is an angel) and (if Charlotte is an angel then Charlotte is a devil)) and Chloe is an angel and (Charlotte is an angel if and only if (Ava is a devil or Jack is a devil or Jack is an angel))." So who is an angel and who is a devil? (Format your answer like: "Ava is a angel/devil, Charlotte is a angel/devil, Jack is a angel/devil, and Chloe is a angel/devil")',
        "Ava is a devil, Charlotte is an angel, Jack is a devil, and Chloe is a devil.",
    ),
    (
        'A very special island is inhabited only by heros and villains. Heros always tell the truth, and villains always lie. You meet 4 inhabitants: Aria, Jack, William, and Aiden. "Jack is a hero," Aria declared. Jack told you that Jack is a hero or ((Aria is a villain if and only if William is a villain) or (Aiden is a villain if and only if Jack is a hero) or (William is a hero) or (if William is a villain then Aria is a hero)) or Aiden is a villain or (it is not the case that (Jack is a hero and William is a hero and Aria is a hero)). "Aria is a villain," William claimed. According to Aiden, "it is not the case that (Aria is a villain if and only if (Jack is a hero))". So who is a hero and who is a villain? (Format your answer like: "Aria is a hero/villain, Jack is a hero/villain, William is a hero/villain, and Aiden is a hero/villain")',
        "Aria is a hero, Jack is a hero, William is a villain, and Aiden is a hero.",
    ),
]


def _normalize_knights_knaves_answer(answer: str) -> set[tuple[str, str]]:
    """Convert an answer string into a normalized set of (name, role) tuples."""
    answer = (
        answer.lower()
        .strip()
        .replace(".", " ")
        .replace(",", " ")
        .replace(")", " ")
        .replace("(", " ")
    )
    parts = answer.replace(" and ", " ").split()

    assignments: set[tuple[str, str]] = set()
    current_name: str | None = None

    for part in parts:
        if part in ("is", "a", "an"):
            continue
        if part in _VALID_ROLES:
            if current_name is not None:
                assignments.add((current_name, part))
                current_name = None
        else:
            current_name = part

    return assignments


def _score_knights_knaves(
    response: str, expected: str, metadata: dict[str, Any]
) -> float:
    """Score a knights and knaves response with partial credit for correct assignments."""
    if not response.strip():
        return 0.0

    try:
        oracle = _normalize_knights_knaves_answer(expected)
        extracted = _normalize_knights_knaves_answer(response)

        if oracle == extracted:
            return 1.0

        if len(oracle) == len(extracted):
            matching = len(oracle.intersection(extracted))
            if matching > 0:
                return 0.3 + (0.7 * matching / len(oracle))
    except Exception:
        pass

    return 0.0


def _build_knights_knaves_tasks() -> list[tuple[str, str, dict[str, Any]]]:
    """Build knights and knaves tasks as (question, answer, metadata) triples."""
    tasks: list[tuple[str, str, dict[str, Any]]] = []
    for index, (question, answer) in enumerate(_KNIGHTS_KNAVES_TASKS):
        metadata = {"source_dataset": "knights_knaves", "source_index": index}
        tasks.append((question, answer, metadata))
    return tasks


# ──────────────────────────────────────────────────
# Futoshiki challenge
# ──────────────────────────────────────────────────

_FUTOSHIKI_QUESTION_TEMPLATE = """\
Solve the following {size}x{size} Futoshiki puzzle:

{puzzle}

Ensure your answer follows the same format as the puzzle above, \
just replace blanks (_) with the correct value for the cell.
Use < and > for horizontal constraints. \
Use \u2227 and \u2228 for vertical constraints.
Remember, in Futoshiki each row and column must contain each number \
from 1 to {size} exactly once."""


def _format_futoshiki_grid(
    grid: list[list[int]],
    constraints: list[tuple[int, int, int, int, str]],
) -> str:
    """Format a Futoshiki grid with constraints as a display string."""
    n = len(grid)
    constraint_map: dict[tuple[int, int, int, int], str] = {
        (r1, c1, r2, c2): sign for r1, c1, r2, c2, sign in constraints
    }

    result_lines: list[str] = []
    for r in range(n):
        # Data row.
        row_parts: list[str] = []
        for c in range(n):
            row_parts.append(str(grid[r][c]) if grid[r][c] != 0 else "_")
            if c < n - 1:
                sign = constraint_map.get((r, c, r, c + 1))
                row_parts.append(sign if sign else " ")
        result_lines.append(" ".join(row_parts))

        # Vertical constraint row.
        if r < n - 1:
            vert_parts: list[str] = []
            for c in range(n):
                sign = constraint_map.get((r, c, r + 1, c))
                if sign == "<":
                    vert_parts.append("\u2227")
                elif sign == ">":
                    vert_parts.append("\u2228")
                else:
                    vert_parts.append(" ")
                if c < n - 1:
                    vert_parts.append(" ")
            result_lines.append(" ".join(vert_parts))

    return "\n".join(result_lines)


# Hand-verified futoshiki tasks: (puzzle_grid, constraints, solution_grid).
# Puzzle grids use 0 for blank cells. Constraints are (r1, c1, r2, c2, sign) tuples.
_FUTOSHIKI_TASKS: list[
    tuple[list[list[int]], list[tuple[int, int, int, int, str]], list[list[int]]]
] = [
    (
        [
            [0, 0, 0, 0, 1, 2],
            [0, 4, 0, 6, 0, 0],
            [0, 0, 0, 4, 0, 0],
            [0, 5, 3, 0, 0, 0],
            [0, 0, 0, 0, 2, 4],
            [3, 1, 0, 5, 0, 0],
        ],
        [
            (1, 4, 1, 5, ">"),
            (4, 1, 4, 2, "<"),
        ],
        [
            [5, 6, 4, 3, 1, 2],
            [2, 4, 1, 6, 5, 3],
            [1, 2, 6, 4, 3, 5],
            [4, 5, 3, 2, 6, 1],
            [6, 3, 5, 1, 2, 4],
            [3, 1, 2, 5, 4, 6],
        ],
    ),
    (
        [
            [5, 0, 0, 4, 0, 2],
            [0, 5, 0, 0, 0, 0],
            [0, 4, 1, 2, 0, 6],
            [4, 0, 0, 0, 0, 0],
            [0, 0, 3, 0, 0, 4],
            [0, 0, 0, 0, 1, 0],
        ],
        [
            (0, 3, 0, 4, ">"),
            (2, 3, 3, 3, "<"),
            (3, 0, 3, 1, ">"),
            (4, 1, 5, 1, "<"),
            (4, 2, 4, 3, "<"),
        ],
        [
            [5, 1, 6, 4, 3, 2],
            [6, 5, 2, 1, 4, 3],
            [3, 4, 1, 2, 5, 6],
            [4, 3, 5, 6, 2, 1],
            [1, 2, 3, 5, 6, 4],
            [2, 6, 4, 3, 1, 5],
        ],
    ),
    (
        [
            [0, 0, 0, 3, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 4],
            [3, 0, 0, 0, 0],
        ],
        [
            (0, 0, 0, 1, "<"),
            (0, 2, 1, 2, ">"),
            (0, 3, 0, 4, ">"),
            (1, 0, 1, 1, ">"),
            (2, 0, 2, 1, "<"),
            (2, 0, 3, 0, ">"),
            (2, 1, 3, 1, ">"),
            (3, 1, 4, 1, ">"),
            (3, 2, 3, 3, "<"),
            (4, 2, 4, 3, "<"),
        ],
        [
            [2, 4, 5, 3, 1],
            [5, 1, 4, 2, 3],
            [4, 5, 3, 1, 2],
            [1, 3, 2, 5, 4],
            [3, 2, 1, 4, 5],
        ],
    ),
    (
        [
            [0, 0, 0, 0, 0, 3, 0, 2, 0],
            [0, 0, 8, 0, 4, 0, 7, 0, 2],
            [6, 0, 0, 7, 8, 0, 0, 0, 0],
            [7, 0, 0, 0, 0, 2, 5, 3, 0],
            [0, 0, 0, 0, 0, 7, 0, 0, 1],
            [0, 0, 0, 0, 2, 0, 0, 0, 8],
            [1, 5, 0, 9, 7, 4, 3, 0, 0],
            [2, 3, 5, 0, 0, 0, 0, 0, 0],
            [0, 2, 9, 0, 3, 0, 0, 0, 0],
        ],
        [
            (0, 0, 1, 0, "<"),
            (0, 1, 0, 2, ">"),
            (0, 2, 0, 3, ">"),
            (0, 2, 1, 2, "<"),
            (0, 4, 0, 5, ">"),
            (0, 5, 0, 6, "<"),
            (1, 0, 1, 1, ">"),
            (1, 1, 1, 2, "<"),
            (1, 2, 1, 3, ">"),
            (1, 2, 2, 2, ">"),
            (1, 3, 2, 3, "<"),
            (1, 4, 2, 4, "<"),
            (1, 5, 1, 6, "<"),
            (1, 6, 1, 7, ">"),
            (1, 7, 2, 7, "<"),
            (2, 1, 3, 1, "<"),
            (2, 2, 2, 3, "<"),
            (2, 2, 3, 2, "<"),
            (2, 3, 3, 3, "<"),
            (2, 4, 3, 4, ">"),
            (2, 7, 3, 7, ">"),
            (3, 1, 3, 2, ">"),
            (3, 5, 4, 5, "<"),
            (3, 8, 4, 8, ">"),
            (4, 0, 4, 1, "<"),
            (4, 1, 4, 2, ">"),
            (4, 2, 4, 3, ">"),
            (4, 4, 4, 5, "<"),
            (4, 4, 5, 4, ">"),
            (4, 5, 4, 6, ">"),
            (4, 8, 5, 8, "<"),
            (5, 2, 6, 2, ">"),
            (5, 6, 5, 7, "<"),
            (6, 0, 6, 1, "<"),
            (6, 1, 6, 2, ">"),
            (6, 1, 7, 1, ">"),
            (6, 2, 6, 3, "<"),
            (6, 3, 7, 3, ">"),
            (6, 5, 6, 6, ">"),
            (6, 5, 7, 5, ">"),
            (6, 7, 7, 7, ">"),
            (6, 8, 7, 8, ">"),
            (7, 2, 8, 2, "<"),
            (7, 4, 8, 4, ">"),
            (7, 6, 7, 7, ">"),
            (8, 0, 8, 1, ">"),
            (8, 1, 8, 2, "<"),
            (8, 3, 8, 4, ">"),
            (8, 6, 8, 7, ">"),
        ],
        [
            [4, 8, 7, 1, 6, 3, 9, 2, 5],
            [9, 1, 8, 3, 4, 6, 7, 5, 2],
            [6, 4, 1, 7, 8, 5, 2, 9, 3],
            [7, 6, 4, 8, 1, 2, 5, 3, 9],
            [8, 9, 3, 2, 5, 7, 4, 6, 1],
            [3, 7, 6, 5, 2, 9, 1, 4, 8],
            [1, 5, 2, 9, 7, 4, 3, 8, 6],
            [2, 3, 5, 6, 9, 1, 8, 7, 4],
            [5, 2, 9, 4, 3, 8, 6, 1, 7],
        ],
    ),
    (
        [
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [2, 4, 0, 0],
            [0, 1, 0, 3],
        ],
        [
            (0, 0, 1, 0, "<"),
            (0, 1, 1, 1, ">"),
            (2, 1, 3, 1, ">"),
        ],
        [
            [1, 3, 4, 2],
            [3, 2, 1, 4],
            [2, 4, 3, 1],
            [4, 1, 2, 3],
        ],
    ),
    (
        [
            [0, 0, 5, 0, 0, 0, 1, 3],
            [8, 0, 0, 4, 6, 0, 0, 0],
            [0, 8, 0, 0, 0, 0, 0, 4],
            [7, 0, 0, 1, 0, 6, 0, 0],
            [0, 4, 0, 6, 0, 0, 5, 0],
            [0, 0, 0, 5, 0, 0, 0, 2],
            [0, 0, 3, 0, 1, 7, 0, 0],
            [0, 0, 0, 0, 7, 3, 0, 8],
        ],
        [
            (1, 1, 1, 2, "<"),
            (1, 2, 2, 2, ">"),
            (1, 5, 1, 6, ">"),
            (1, 6, 1, 7, ">"),
            (2, 5, 2, 6, "<"),
            (3, 0, 4, 0, ">"),
            (3, 5, 3, 6, ">"),
            (3, 7, 4, 7, "<"),
            (4, 2, 4, 3, ">"),
            (5, 2, 6, 2, ">"),
            (6, 0, 7, 0, "<"),
        ],
        [
            [2, 6, 5, 7, 4, 8, 1, 3],
            [8, 2, 7, 4, 6, 5, 3, 1],
            [6, 8, 1, 3, 5, 2, 7, 4],
            [7, 3, 2, 1, 8, 6, 4, 5],
            [3, 4, 8, 6, 2, 1, 5, 7],
            [1, 7, 6, 5, 3, 4, 8, 2],
            [4, 5, 3, 8, 1, 7, 2, 6],
            [5, 1, 4, 2, 7, 3, 6, 8],
        ],
    ),
    (
        [
            [6, 0, 0, 3, 0, 4],
            [0, 0, 0, 0, 3, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 3, 6, 0, 0, 0],
            [0, 6, 5, 4, 0, 0],
        ],
        [
            (0, 2, 1, 2, ">"),
            (1, 5, 2, 5, ">"),
            (3, 0, 4, 0, "<"),
            (3, 2, 4, 2, "<"),
            (3, 3, 4, 3, "<"),
            (3, 4, 3, 5, "<"),
            (4, 4, 5, 4, "<"),
            (5, 0, 5, 1, "<"),
            (5, 4, 5, 5, ">"),
        ],
        [
            [6, 1, 2, 3, 5, 4],
            [2, 4, 1, 6, 3, 5],
            [5, 2, 4, 1, 6, 3],
            [1, 5, 3, 2, 4, 6],
            [4, 3, 6, 5, 1, 2],
            [3, 6, 5, 4, 2, 1],
        ],
    ),
    (
        [
            [0, 0, 2, 0],
            [0, 4, 0, 0],
            [3, 0, 0, 0],
            [1, 0, 0, 0],
        ],
        [
            (1, 0, 1, 1, "<"),
            (1, 1, 1, 2, ">"),
            (1, 2, 1, 3, "<"),
            (2, 1, 2, 2, "<"),
            (2, 3, 3, 3, "<"),
        ],
        [
            [4, 3, 2, 1],
            [2, 4, 1, 3],
            [3, 1, 4, 2],
            [1, 2, 3, 4],
        ],
    ),
    (
        [
            [0, 0, 2, 6, 0, 4, 0, 0, 0],
            [0, 0, 0, 3, 0, 8, 5, 1, 4],
            [0, 5, 8, 0, 0, 0, 6, 0, 1],
            [9, 4, 1, 0, 0, 0, 0, 7, 5],
            [0, 0, 0, 0, 0, 9, 1, 0, 7],
            [7, 2, 0, 0, 0, 0, 0, 3, 9],
            [0, 1, 0, 7, 9, 6, 0, 0, 0],
            [5, 8, 0, 0, 0, 0, 0, 0, 3],
            [0, 0, 9, 0, 0, 5, 0, 0, 0],
        ],
        [
            (0, 2, 1, 2, "<"),
            (0, 4, 1, 4, ">"),
            (0, 6, 0, 7, ">"),
            (1, 5, 1, 6, ">"),
            (1, 8, 2, 8, ">"),
            (2, 0, 3, 0, "<"),
            (2, 1, 2, 2, "<"),
            (2, 5, 3, 5, ">"),
            (2, 8, 3, 8, "<"),
            (4, 3, 4, 4, ">"),
            (6, 3, 6, 4, "<"),
            (6, 5, 6, 6, ">"),
            (7, 6, 7, 7, "<"),
        ],
        [
            [1, 7, 2, 6, 3, 4, 9, 5, 8],
            [6, 9, 7, 3, 2, 8, 5, 1, 4],
            [4, 5, 8, 2, 7, 3, 6, 9, 1],
            [9, 4, 1, 8, 6, 2, 3, 7, 5],
            [8, 6, 3, 5, 4, 9, 1, 2, 7],
            [7, 2, 6, 4, 5, 1, 8, 3, 9],
            [3, 1, 5, 7, 9, 6, 4, 8, 2],
            [5, 8, 4, 9, 1, 7, 2, 6, 3],
            [2, 3, 9, 1, 8, 5, 7, 4, 6],
        ],
    ),
    (
        [
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 6, 4],
            [0, 0, 0, 0, 2, 0],
            [0, 0, 3, 4, 0, 0],
            [0, 4, 2, 0, 0, 3],
            [0, 5, 4, 6, 0, 0],
        ],
        [
            (2, 4, 3, 4, ">"),
            (3, 5, 4, 5, ">"),
            (4, 1, 4, 2, ">"),
        ],
        [
            [1, 3, 6, 5, 4, 2],
            [3, 1, 5, 2, 6, 4],
            [4, 6, 1, 3, 2, 5],
            [5, 2, 3, 4, 1, 6],
            [6, 4, 2, 1, 5, 3],
            [2, 5, 4, 6, 3, 1],
        ],
    ),
    (
        [
            [2, 0, 3, 0, 0, 6, 1],
            [0, 5, 7, 0, 0, 0, 0],
            [0, 0, 6, 0, 2, 1, 0],
            [0, 0, 0, 1, 0, 0, 2],
            [4, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 3, 5, 4, 0],
            [6, 0, 0, 0, 0, 0, 3],
        ],
        [
            (0, 2, 0, 3, "<"),
            (0, 3, 0, 4, ">"),
            (1, 3, 2, 3, "<"),
            (1, 5, 2, 5, ">"),
            (2, 0, 2, 1, ">"),
            (2, 3, 2, 4, ">"),
            (2, 3, 3, 3, ">"),
            (3, 3, 3, 4, "<"),
            (3, 4, 3, 5, ">"),
            (4, 1, 5, 1, "<"),
            (5, 5, 5, 6, "<"),
        ],
        [
            [2, 7, 3, 5, 4, 6, 1],
            [3, 5, 7, 4, 1, 2, 6],
            [5, 3, 6, 7, 2, 1, 4],
            [7, 4, 5, 1, 6, 3, 2],
            [4, 2, 1, 6, 3, 7, 5],
            [1, 6, 2, 3, 5, 4, 7],
            [6, 1, 4, 2, 7, 5, 3],
        ],
    ),
    (
        [
            [2, 1, 0, 0, 9, 0, 0, 0, 3],
            [0, 0, 6, 2, 8, 0, 0, 3, 1],
            [0, 0, 0, 8, 5, 0, 0, 0, 4],
            [9, 3, 0, 0, 0, 8, 0, 1, 5],
            [0, 0, 0, 7, 0, 3, 0, 0, 0],
            [0, 9, 0, 0, 0, 0, 0, 0, 8],
            [4, 5, 0, 9, 0, 0, 1, 0, 0],
            [0, 0, 5, 0, 0, 0, 0, 4, 0],
            [0, 7, 3, 0, 1, 5, 8, 0, 0],
        ],
        [
            (0, 1, 1, 1, "<"),
            (0, 2, 0, 3, ">"),
            (0, 2, 1, 2, ">"),
            (0, 3, 0, 4, "<"),
            (0, 3, 1, 3, ">"),
            (0, 4, 1, 4, ">"),
            (1, 0, 2, 0, ">"),
            (1, 1, 1, 2, "<"),
            (1, 2, 2, 2, "<"),
            (1, 4, 1, 5, "<"),
            (1, 6, 1, 7, ">"),
            (1, 7, 1, 8, ">"),
            (1, 7, 2, 7, "<"),
            (2, 0, 2, 1, "<"),
            (2, 1, 2, 2, "<"),
            (2, 1, 3, 1, ">"),
            (2, 2, 2, 3, ">"),
            (2, 2, 3, 2, ">"),
            (2, 4, 3, 4, "<"),
            (2, 5, 2, 6, "<"),
            (2, 5, 3, 5, "<"),
            (2, 6, 2, 7, "<"),
            (3, 6, 3, 7, ">"),
            (4, 0, 4, 1, ">"),
            (4, 1, 4, 2, ">"),
            (4, 3, 5, 3, ">"),
            (4, 4, 4, 5, ">"),
            (4, 4, 5, 4, ">"),
            (4, 5, 5, 5, "<"),
            (5, 0, 5, 1, "<"),
            (5, 0, 6, 0, "<"),
            (5, 1, 5, 2, ">"),
            (5, 2, 5, 3, ">"),
            (5, 2, 6, 2, "<"),
            (5, 7, 5, 8, "<"),
            (6, 0, 7, 0, "<"),
            (6, 3, 7, 3, ">"),
            (7, 0, 8, 0, ">"),
            (7, 2, 8, 2, ">"),
            (7, 3, 7, 4, ">"),
            (7, 4, 7, 5, ">"),
            (7, 4, 8, 4, ">"),
            (8, 1, 8, 2, ">"),
            (8, 3, 8, 4, ">"),
            (8, 6, 8, 7, "<"),
        ],
        [
            [2, 1, 7, 5, 9, 4, 6, 8, 3],
            [5, 4, 6, 2, 8, 9, 7, 3, 1],
            [1, 6, 9, 8, 5, 2, 3, 7, 4],
            [9, 3, 4, 6, 7, 8, 2, 1, 5],
            [8, 2, 1, 7, 6, 3, 4, 5, 9],
            [3, 9, 2, 1, 4, 7, 5, 6, 8],
            [4, 5, 8, 9, 3, 6, 1, 2, 7],
            [7, 8, 5, 3, 2, 1, 9, 4, 6],
            [6, 7, 3, 4, 1, 5, 8, 9, 2],
        ],
    ),
    (
        [
            [0, 3, 5, 0, 0, 2],
            [2, 0, 0, 0, 6, 4],
            [0, 0, 6, 4, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 6, 0, 3, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ],
        [
            (0, 1, 0, 2, "<"),
            (0, 1, 1, 1, ">"),
            (1, 2, 1, 3, "<"),
            (1, 3, 2, 3, ">"),
            (1, 5, 2, 5, ">"),
            (2, 5, 3, 5, "<"),
            (3, 1, 4, 1, "<"),
            (3, 2, 3, 3, ">"),
            (3, 3, 3, 4, "<"),
            (4, 0, 4, 1, "<"),
            (4, 0, 5, 0, ">"),
            (4, 1, 4, 2, ">"),
            (4, 2, 4, 3, "<"),
            (4, 3, 4, 4, ">"),
            (4, 4, 5, 4, "<"),
            (5, 1, 5, 2, ">"),
        ],
        [
            [1, 3, 5, 6, 4, 2],
            [2, 1, 3, 5, 6, 4],
            [5, 2, 6, 4, 3, 1],
            [6, 5, 4, 1, 2, 3],
            [4, 6, 2, 3, 1, 5],
            [3, 4, 1, 2, 5, 6],
        ],
    ),
    (
        [
            [6, 0, 7, 2, 0, 8, 4, 1, 0],
            [0, 8, 0, 6, 0, 0, 1, 0, 0],
            [0, 0, 6, 0, 0, 0, 9, 5, 0],
            [5, 0, 0, 0, 0, 0, 7, 4, 2],
            [0, 0, 5, 1, 8, 0, 0, 7, 0],
            [0, 7, 8, 9, 0, 1, 0, 0, 0],
            [0, 0, 4, 0, 1, 6, 0, 0, 9],
            [4, 3, 1, 0, 2, 0, 0, 0, 0],
            [1, 4, 0, 0, 0, 5, 0, 3, 0],
        ],
        [
            (0, 0, 1, 0, ">"),
            (0, 5, 1, 5, ">"),
            (0, 7, 1, 7, "<"),
            (1, 1, 2, 1, ">"),
            (1, 6, 2, 6, "<"),
            (2, 0, 3, 0, ">"),
            (2, 5, 2, 6, "<"),
            (3, 1, 3, 2, "<"),
            (4, 2, 5, 2, "<"),
            (4, 4, 4, 5, ">"),
            (4, 8, 5, 8, "<"),
            (6, 6, 6, 7, "<"),
            (7, 1, 7, 2, ">"),
            (7, 3, 7, 4, ">"),
            (7, 3, 8, 3, "<"),
            (7, 6, 7, 7, ">"),
            (7, 7, 7, 8, "<"),
        ],
        [
            [6, 9, 7, 2, 5, 8, 4, 1, 3],
            [2, 8, 3, 6, 7, 4, 1, 9, 5],
            [8, 2, 6, 4, 3, 7, 9, 5, 1],
            [5, 1, 9, 8, 6, 3, 7, 4, 2],
            [9, 6, 5, 1, 8, 2, 3, 7, 4],
            [3, 7, 8, 9, 4, 1, 5, 2, 6],
            [7, 5, 4, 3, 1, 6, 2, 8, 9],
            [4, 3, 1, 5, 2, 9, 8, 6, 7],
            [1, 4, 2, 7, 9, 5, 6, 3, 8],
        ],
    ),
    (
        [
            [9, 3, 0, 8, 4, 2, 6, 0, 0],
            [3, 0, 0, 0, 2, 4, 0, 7, 0],
            [2, 0, 9, 0, 8, 0, 0, 6, 3],
            [0, 2, 0, 0, 6, 8, 0, 9, 0],
            [0, 7, 0, 0, 0, 0, 0, 3, 8],
            [4, 9, 2, 3, 0, 0, 0, 0, 5],
            [0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 9, 1, 3, 2, 0],
            [8, 0, 0, 2, 1, 0, 7, 0, 9],
        ],
        [
            (0, 5, 1, 5, "<"),
            (0, 7, 1, 7, "<"),
            (1, 0, 1, 1, "<"),
            (1, 3, 1, 4, "<"),
            (1, 5, 1, 6, "<"),
            (2, 2, 3, 2, ">"),
            (3, 5, 3, 6, ">"),
            (4, 3, 4, 4, "<"),
            (4, 5, 5, 5, ">"),
            (4, 6, 5, 6, "<"),
            (5, 0, 5, 1, "<"),
            (5, 2, 6, 2, ">"),
            (6, 4, 6, 5, "<"),
            (7, 6, 7, 7, ">"),
            (7, 7, 7, 8, "<"),
            (8, 3, 8, 4, ">"),
        ],
        [
            [9, 3, 7, 8, 4, 2, 6, 5, 1],
            [3, 5, 8, 1, 2, 4, 9, 7, 6],
            [2, 1, 9, 7, 8, 5, 4, 6, 3],
            [7, 2, 3, 5, 6, 8, 1, 9, 4],
            [1, 7, 6, 4, 5, 9, 2, 3, 8],
            [4, 9, 2, 3, 7, 6, 8, 1, 5],
            [6, 4, 1, 9, 3, 7, 5, 8, 2],
            [5, 8, 4, 6, 9, 1, 3, 2, 7],
            [8, 6, 5, 2, 1, 3, 7, 4, 9],
        ],
    ),
    (
        [
            [0, 0, 0, 7, 4, 0, 0],
            [0, 3, 0, 0, 0, 0, 1],
            [5, 0, 0, 0, 0, 6, 7],
            [0, 0, 7, 0, 3, 0, 0],
            [0, 0, 4, 0, 7, 0, 2],
            [3, 0, 0, 1, 0, 0, 0],
            [6, 0, 3, 5, 0, 0, 0],
        ],
        [
            (0, 1, 0, 2, ">"),
            (1, 6, 2, 6, "<"),
            (2, 3, 2, 4, ">"),
            (3, 2, 3, 3, ">"),
            (3, 5, 3, 6, "<"),
            (4, 0, 4, 1, "<"),
            (5, 6, 6, 6, ">"),
            (6, 3, 6, 4, ">"),
        ],
        [
            [2, 6, 5, 7, 4, 1, 3],
            [7, 3, 6, 4, 5, 2, 1],
            [5, 4, 1, 3, 2, 6, 7],
            [4, 1, 7, 2, 3, 5, 6],
            [1, 5, 4, 6, 7, 3, 2],
            [3, 7, 2, 1, 6, 4, 5],
            [6, 2, 3, 5, 1, 7, 4],
        ],
    ),
]


def _score_futoshiki(response: str, expected: str, metadata: dict[str, Any]) -> float:
    """Score a futoshiki response using exact match with fallback to numeric comparison."""
    if not response.strip():
        return 0.0

    solution: list[list[int]] = metadata["solution"]
    board_size: int = metadata["board_size"]
    answer_string: str = metadata["answer_string"]

    # Exact match after stripping trailing whitespace per line.
    response_stripped = "\n".join(line.rstrip() for line in response.split("\n"))
    answer_stripped = "\n".join(line.rstrip() for line in answer_string.split("\n"))

    if response_stripped == answer_stripped:
        return 1.0

    # Numeric comparison: extract digit sequences and compare with solution.
    row = 0
    num_matching = 0
    for line in response.split("\n"):
        if row >= board_size:
            break
        numbers = [int(c) for c in line if c in "123456789"]
        if len(numbers) != board_size:
            continue
        for actual, expected_val in zip(solution[row], numbers):
            if actual == expected_val:
                num_matching += 1
        row += 1

    total_cells = board_size * board_size
    reward = (num_matching / total_cells) * 0.9

    # Length penalty for excessively long responses.
    if len(response) > len(answer_string):
        reward *= len(answer_string) / len(response)

    return reward


def _build_futoshiki_tasks() -> list[tuple[str, str, dict[str, Any]]]:
    """Build futoshiki tasks as (question, answer, metadata) triples."""
    tasks: list[tuple[str, str, dict[str, Any]]] = []
    for index, (puzzle, constraints, solution) in enumerate(_FUTOSHIKI_TASKS):
        board_size = len(solution)
        puzzle_grid = _format_futoshiki_grid(puzzle, constraints)
        answer_grid = _format_futoshiki_grid(solution, constraints)
        question = _FUTOSHIKI_QUESTION_TEMPLATE.format(
            size=board_size, puzzle=puzzle_grid
        )
        metadata = {
            "source_dataset": "futoshiki",
            "source_index": index,
            "solution": solution,
            "board_size": board_size,
            "answer_string": answer_grid,
        }
        tasks.append((question, answer_grid, metadata))
    return tasks


# ──────────────────────────────────────────────────
# Jugs challenge
# ──────────────────────────────────────────────────

_JUGS_QUESTION_TEMPLATE = """You are a police officer. A maniac has planted a bomb next to a public fountain.

To defuse the bomb, you must solve a puzzle. The puzzle is solved when you fill any of the available jugs with the target amount of water.

You have three move types: 'fill', 'empty' and 'pour'.

To fill Jug A, you 'fill A'.
To empty Jug B, you 'empty B'.
To pour the contents of Jug A into Jug B, you 'pour A->B'.
All jugs are empty to begin with.

The empty jugs hold this many litres of water: {jug_spec}
And your target is: {target} litres.

How do you defuse the bomb?

Reply as a JSON-parsable list of moves which result in any of the jugs being filled with the target amount.
"""

# Hand-verified jugs tasks: (jug_capacities, target, reference_moves).
_JUGS_TASKS: list[tuple[list[int], int, list[str]]] = [
    (
        [5, 11, 11],
        8,
        [
            "fill B",
            "pour B->A",
            "empty A",
            "pour B->A",
            "empty A",
            "pour B->A",
            "fill B",
            "pour B->A",
            "empty A",
            "pour B->A",
            "empty A",
            "pour B->A",
            "fill B",
            "pour B->A",
        ],
    ),
    (
        [13, 13, 8],
        12,
        [
            "fill A",
            "pour A->C",
            "fill B",
            "empty C",
            "pour A->C",
            "fill A",
            "pour A->C",
            "empty C",
            "pour A->C",
            "empty C",
            "pour A->C",
            "fill A",
            "pour A->C",
            "empty C",
            "pour A->C",
            "pour B->C",
        ],
    ),
    (
        [4, 13, 4],
        2,
        [
            "fill B",
            "pour B->A",
            "empty A",
            "pour B->A",
            "empty A",
            "pour B->A",
            "empty A",
            "pour B->A",
            "fill B",
            "pour B->A",
            "empty A",
            "pour B->A",
            "pour B->C",
        ],
    ),
    (
        [9, 11, 9],
        3,
        [
            "fill A",
            "pour A->B",
            "fill A",
            "pour A->B",
            "empty B",
            "pour A->B",
            "fill A",
            "pour A->B",
            "empty B",
            "pour A->B",
            "fill A",
            "pour A->B",
        ],
    ),
    (
        [4, 13, 13],
        2,
        [
            "fill B",
            "pour B->A",
            "empty A",
            "pour B->A",
            "empty A",
            "pour B->A",
            "empty A",
            "pour B->A",
            "fill B",
            "pour B->A",
            "empty A",
            "pour B->A",
            "empty A",
            "pour B->A",
        ],
    ),
    (
        [6, 11, 6],
        8,
        [
            "fill A",
            "pour A->B",
            "fill A",
            "pour A->B",
            "empty B",
            "pour A->B",
            "fill A",
            "pour A->B",
            "fill A",
            "pour A->B",
            "empty B",
            "pour A->B",
            "fill A",
            "pour A->B",
        ],
    ),
    (
        [13, 7, 13],
        9,
        [
            "fill B",
            "pour B->A",
            "fill B",
            "pour B->A",
            "empty A",
            "pour B->A",
            "fill B",
            "pour B->A",
            "fill B",
            "pour B->A",
            "pour B->C",
            "pour A->B",
            "pour B->C",
        ],
    ),
    (
        [13, 5, 5],
        9,
        [
            "fill A",
            "pour A->B",
            "pour A->C",
            "empty B",
            "pour A->B",
            "fill A",
            "pour A->B",
            "empty B",
            "pour A->B",
            "empty B",
            "pour A->B",
            "empty B",
            "pour A->B",
            "fill A",
            "pour A->B",
        ],
    ),
    (
        [7, 7, 13],
        9,
        [
            "fill A",
            "pour A->C",
            "fill A",
            "pour A->C",
            "fill B",
            "empty C",
            "pour A->C",
            "fill A",
            "pour A->C",
            "fill A",
            "pour A->C",
            "empty C",
            "pour A->C",
            "pour B->C",
        ],
    ),
    (
        [3, 3, 11],
        4,
        [
            "fill A",
            "pour A->C",
            "fill A",
            "pour A->C",
            "fill A",
            "pour A->C",
            "fill A",
            "pour A->C",
            "fill B",
            "empty C",
            "pour A->C",
            "pour B->C",
        ],
    ),
    (
        [13, 7, 7],
        3,
        [
            "fill B",
            "pour B->A",
            "fill B",
            "pour B->A",
            "empty A",
            "pour B->A",
            "fill B",
            "pour B->A",
            "fill B",
            "pour B->A",
            "empty A",
            "pour B->A",
            "fill B",
            "pour B->A",
            "fill B",
            "pour B->A",
        ],
    ),
    (
        [12, 13, 13],
        6,
        [
            "fill B",
            "pour B->A",
            "empty A",
            "pour B->A",
            "fill B",
            "pour B->A",
            "empty A",
            "pour B->A",
            "fill B",
            "pour B->A",
            "empty A",
            "pour B->A",
            "fill B",
            "pour B->A",
            "empty A",
            "pour B->A",
            "fill B",
            "pour B->A",
            "empty A",
            "pour B->A",
            "fill B",
            "pour B->A",
        ],
    ),
    (
        [13, 13, 12],
        9,
        [
            "fill C",
            "pour C->A",
            "fill C",
            "pour C->A",
            "empty A",
            "pour C->A",
            "fill C",
            "pour C->A",
            "pour C->B",
            "pour A->C",
            "pour C->B",
        ],
    ),
    (
        [13, 10, 3],
        8,
        [
            "fill B",
            "pour B->C",
            "empty C",
            "pour B->C",
            "empty C",
            "pour B->C",
            "empty C",
            "pour B->C",
            "fill B",
            "pour B->C",
        ],
    ),
    (
        [9, 9, 11],
        8,
        [
            "fill C",
            "pour C->A",
            "empty A",
            "pour C->A",
            "fill C",
            "pour C->A",
            "empty A",
            "pour C->A",
            "fill C",
            "pour C->A",
            "pour C->B",
            "fill C",
            "pour C->B",
        ],
    ),
    (
        [4, 13, 13],
        11,
        [
            "fill A",
            "pour A->B",
            "fill A",
            "pour A->B",
            "fill A",
            "pour A->B",
            "fill A",
            "pour A->B",
            "pour A->C",
            "fill A",
            "pour A->C",
            "fill A",
            "pour A->C",
        ],
    ),
]


def _simulate_jugs(capacities: list[int], target: int, moves: list[str]) -> bool:
    """Simulate a sequence of jug moves and check if any jug reaches the target."""
    n = len(capacities)
    jug_map = {chr(ord("A") + i): i for i in range(n)}
    state = [0] * n

    for move in moves:
        tokens = move.split()
        action = tokens[0].lower()

        if action == "fill":
            idx = jug_map.get(tokens[1].upper())
            if idx is None:
                return False
            state[idx] = capacities[idx]
        elif action == "empty":
            idx = jug_map.get(tokens[1].upper())
            if idx is None:
                return False
            state[idx] = 0
        elif action == "pour":
            parts = tokens[1].split("->")
            if len(parts) != 2:
                return False
            source = jug_map.get(parts[0].upper())
            dest = jug_map.get(parts[1].upper())
            if source is None or dest is None:
                return False
            amount = min(state[source], capacities[dest] - state[dest])
            state[source] -= amount
            state[dest] += amount
        else:
            return False

    return any(w == target for w in state)


def _extract_json_moves(response: str) -> list[str] | None:
    """Extract a JSON array of move strings from a model response."""
    cleaned = response.strip()

    # Try parsing the entire response as JSON.
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list) and all(isinstance(m, str) for m in parsed):
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass

    # Try extracting a JSON array from within the response.
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start != -1 and end > start:
        try:
            parsed = json.loads(cleaned[start : end + 1])
            if isinstance(parsed, list) and all(isinstance(m, str) for m in parsed):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass

    return None


def _score_jugs(response: str, expected: str, metadata: dict[str, Any]) -> float:
    """Score a jugs challenge response by simulating the proposed moves."""
    if not response.strip():
        return 0.0

    capacities: list[int] = metadata["jug_capacities"]
    target: int = metadata["target"]

    moves = _extract_json_moves(response)
    if moves is None:
        return 0.0

    # JSON-parseable but might not solve the puzzle.
    try:
        if _simulate_jugs(capacities, target, moves):
            return 1.0
    except (IndexError, KeyError, ValueError):
        return 0.0

    return 0.01


def _build_jugs_tasks() -> list[tuple[str, str, dict[str, Any]]]:
    """Build jugs tasks as (question, answer, metadata) triples."""
    tasks: list[tuple[str, str, dict[str, Any]]] = []
    for index, (capacities, target, reference_moves) in enumerate(_JUGS_TASKS):
        jug_labels = [chr(ord("A") + i) for i in range(len(capacities))]
        jug_spec = ", ".join(
            f"{label}:{cap}" for label, cap in zip(jug_labels, capacities)
        )
        question = _JUGS_QUESTION_TEMPLATE.format(jug_spec=jug_spec, target=target)
        answer = json.dumps(reference_moves)
        metadata = {
            "source_dataset": "jugs",
            "source_index": index,
            "jug_capacities": capacities,
            "target": target,
        }
        tasks.append((question, answer, metadata))
    return tasks


# ──────────────────────────────────────────────────
# Sentence reordering challenge
# ──────────────────────────────────────────────────

_SENTENCE_REORDERING_INSTRUCTIONS = (
    "Restore the correct order of words in the following sentence.\n"
    "Return only the reordered sentence with no explanation.\n"
)

# Hand-verified sentence reordering tasks: (scrambled, answer, alt_answers).
# Alt answers map alternative phrasings to their hand-assigned scores.
_SENTENCE_REORDERING_TASKS: list[tuple[str, str, dict[str, float]]] = [
    (
        "him Woe to goodness! good is name to whose more than him",
        "Woe to him whose good name is more to him than goodness!",
        {
            "Woe to him whose good name is to him more than goodness!": 0.93,
        },
    ),
    (
        "purse, test him Jonah's Jonah's of Captain prepares to So openly. judge ere he the length",
        "So Jonah's Captain prepares to test the length of Jonah's purse, ere he judge him openly.",
        {
            "So Jonah's Captain prepares to judge the length of Jonah's purse, ere he test him openly.": 0.6,
        },
    ),
    (
        "and a way, is passage. mutters; Jonah he down his Not for any put forger,",
        "Not a forger, any way, he mutters; and Jonah is put down for his passage.",
        {},
    ),
    (
        "the kelson main-truck is low? Is not higher the than",
        "Is not the main-truck higher than the kelson is low?",
        {
            "Is the main-truck not higher than the kelson is low?": 0.95,
            "Is not the kelson higher than the main-truck is low?": 0.3,
            "Is the kelson not higher than the main-truck is low?": 0.25,
        },
    ),
    (
        "For he weep as does sinful direct wail and deliverance. not for is, Jonah",
        "For sinful as he is, Jonah does not weep and wail for direct deliverance.",
        {
            "For sinful as he is, Jonah does not wail and weep for direct deliverance.": 0.95,
        },
    ),
    (
        "to him salvation! true, to even Woe be be were false would though who not",
        "Woe to him who would not be true, even though to be false were salvation!",
        {
            "Woe to him who would not even be true, though to be false were salvation!": 0.55,
            "Woe to him who would not be true, though even to be false were salvation!": 0.35,
            "Woe to him who would not be true, though to be false were even salvation!": 0.58,
        },
    ),
    (
        "and the in But contradiction more appals lamp him. that more",
        "But that contradiction in the lamp more and more appals him.",
        {
            "But more and more that contradiction in the lamp appals him.": 0.95,
            "But the contradiction in that lamp more and more appals him.": 0.65,
        },
    ),
    (
        "has pour the God a waters them when brewed who him oil to upon seeks to gale! into Woe",
        "Woe to him who seeks to pour oil upon the waters when God has brewed them into a gale!",
        {
            "Woe to him who seeks to pour oil upon the waters when God has brewed into them a gale!": 0.63,
            "Woe to him who seeks to pour oil into the waters when God has brewed upon them a gale!": 0.35,
            "Woe to him who seeks to pour upon the waters oil when God has brewed them into a gale!": 0.8,
        },
    ),
    (
        "of Then Jonah the Lord prayed the fish's belly. unto out",
        "Then Jonah prayed unto the Lord out of the fish's belly.",
        {
            "Jonah then prayed unto the Lord out of the fish's belly.": 0.95,
            "Then unto the Lord Jonah prayed out of the fish's belly.": 0.8,
        },
    ),
    (
        "will strong not confess himself that suspected; is itself but He suspicion.",
        "He will not confess himself suspected; but that itself is strong suspicion.",
        {
            "He will not confess himself suspected; but that is itself strong suspicion.": 0.95,
            "He will confess himself not suspected; but that itself is strong suspicion.": 0.2,
        },
    ),
    (
        "to. He and him charges it's the thrice sum; usual assented",
        "He charges him thrice the usual sum; and it's assented to.",
        {
            "He charges him the usual sum thrice; and it's assented to.": 0.4,
        },
    ),
    (
        "his attitudes, too known. plainly In God-fugitive all is cringing now the",
        "In all his cringing attitudes, the God-fugitive is now too plainly known.",
        {
            "In all his cringing attitudes, the God-fugitive now is too plainly known.": 0.95,
            "In all his cringing attitudes, the God-fugitive is too plainly now known.": 0.8,
            "In all his cringing attitudes, the God-fugitive is now plainly too known.": 0.25,
            "In all his attitudes, the cringing God-fugitive is now too plainly known.": 0.75,
            "In all his attitudes, the cringing God-fugitive now is too plainly known.": 0.72,
        },
    ),
    (
        "this duty! from charms world whom to him Woe Gospel",
        "Woe to him whom this world charms from Gospel duty!",
        {
            "Woe to him from whom this world charms Gospel duty!": 0.3,
        },
    ),
    (
        "in to mild a people condense. scattered Father and of the voice Mapple authority unassuming rose, ordered",
        "Father Mapple rose, and in a mild voice of unassuming authority ordered the scattered people to condense.",
        {
            "Father Mapple rose, and ordered the scattered people to condense in a mild voice of unassuming authority.": 0.75,
            "Father Mapple rose, and in a voice of mild unassuming authority ordered the scattered people to condense.": 0.85,
            "Father Mapple rose, and in a mild voice of unassuming authority ordered the people scattered to condense.": 0.6,
            "Father Mapple rose, and in a mild voice ordered the scattered people of unassuming authority to condense.": 0.3,
        },
    ),
    (
        "so boldness a only face, Jonah trembles, and summoning more much looks coward. his Frighted to all his the",
        "Frighted Jonah trembles, and summoning all his boldness to his face, only looks so much the more a coward.",
        {
            "Frighted Jonah trembles, and summoning all his boldness to his face, looks only so much the more a coward.": 0.95,
            "Frighted Jonah trembles, and summoning his boldness to all his face, only looks so much the more a coward.": 0.35,
        },
    ),
    (
        "this the And context, with of taken meaning. is full",
        "And taken with the context, this is full of meaning.",
        {
            "And with the context, this is taken full of meaning.": 0.4,
        },
    ),
    (
        "fish's a that What noble belly! thing is in the canticle",
        "What a noble thing is that canticle in the fish's belly!",
        {
            "What a noble canticle is that thing in the fish's belly!": 0.5,
            "What a thing is that noble canticle in the fish's belly!": 0.45,
            "What a noble thing is the canticle in that fish's belly!": 0.7,
        },
    ),
    (
        "Tarshish accounts city the than other no modern By Cadiz. could all been have",
        "By all accounts Tarshish could have been no other city than the modern Cadiz.",
        {
            "By all accounts Tarshish could have been no city other than the modern Cadiz.": 0.95,
            "By all accounts the modern Tarshish could have been no other city than Cadiz.": 0.25,
        },
    ),
    (
        "skulks Tarshish. Joppa, wharves and for seeks He ship that's of the about bound a",
        "He skulks about the wharves of Joppa, and seeks a ship that's bound for Tarshish.",
        {
            "He seeks the wharves of Joppa, and skulks about a ship that's bound for Tarshish.": 0.55,
            "He skulks about a ship that's bound for Joppa, and seeks the wharves of Tarshish.": 0.25,
        },
    ),
    (
        "and for is pardon, here, faithful repentance; but shipmates, And not true clamorous grateful for punishment.",
        "And here, shipmates, is true and faithful repentance; not clamorous for pardon, but grateful for punishment.",
        {
            "And here, shipmates, is faithful and true repentance; not clamorous for pardon, but grateful for punishment.": 0.95,
            "And here, shipmates, is true and faithful repentance; not grateful for pardon, but clamorous for punishment.": 0.3,
            "And here, shipmates, is faithful and true repentance; not grateful for pardon, but clamorous for punishment.": 0.25,
        },
    ),
    (
        "world-wide not from flee God? that Jonah ye to sought shipmates, See then,",
        "See ye not then, shipmates, that Jonah sought to flee world-wide from God?",
        {
            "See ye not then, shipmates, that Jonah sought world-wide to flee from God?": 0.9,
        },
    ),
    (
        "world, to Woe him courts this dishonor! who, in not",
        "Woe to him who, in this world, courts not dishonor!",
        {
            "Woe to him who, in this world, courts dishonor not!": 0.9,
        },
    ),
    (
        "it take Sin but you like do, to heed if of repent Jonah. not;",
        "Sin not; but if you do, take heed to repent of it like Jonah.",
        {
            "Sin not; but if you do, take heed of it to repent like Jonah.": 0.48,
        },
    ),
    (
        "how mob then their Jonah's; him with lot discovered, The they that questions. furiously is",
        "The lot is Jonah's; that discovered, then how furiously they mob him with their questions.",
        {
            "The lot is Jonah's; that discovered, how furiously then they mob him with their questions.": 0.95,
            "The lot is Jonah's; that discovered, then how they furiously mob him with their questions.": 0.75,
            "The lot is Jonah's; that discovered, how they then furiously mob him with their questions.": 0.7,
        },
    ),
    (
        "others the a Paul while preaching woe great himself to is has it, castaway! Pilot as Yea, him who, to",
        "Yea, woe to him who, as the great Pilot Paul has it, while preaching to others is himself a castaway!",
        {
            "Yea, woe to him who, as Paul the great Pilot has it, while preaching to others is himself a castaway!": 0.85,
            "Yea, woe to him who, as the great Pilot Paul has it, while preaching to others himself is a castaway!": 0.9,
            "Yea, woe to him who, as the great Pilot Paul has it, is himself while preaching to others a castaway!": 0.7,
        },
    ),
    (
        "who than him to Woe please appal! to rather to seeks",
        "Woe to him who seeks to please rather than to appal!",
        {
            "Woe to him who seeks rather to please than to appal!": 0.95,
            "Woe to him who rather seeks to please than to appal!": 0.9,
        },
    ),
]


def _score_sentence_reordering(
    response: str, expected: str, metadata: dict[str, Any]
) -> float:
    """Score a sentence reordering response against the canonical and alternative answers."""
    cleaned = response.strip()
    if not cleaned:
        return 0.0

    # Exact match with the canonical answer.
    if cleaned == expected:
        return 1.0

    # Match with a known alternative answer.
    alt_answers: dict[str, float] = metadata["alt_answers"]
    if cleaned in alt_answers:
        return alt_answers[cleaned]

    return 0.0


def _build_sentence_reordering_tasks() -> list[tuple[str, str, dict[str, Any]]]:
    """Build sentence reordering tasks as (question, answer, metadata) triples."""
    tasks: list[tuple[str, str, dict[str, Any]]] = []
    for index, (scrambled, answer, alt_answers) in enumerate(
        _SENTENCE_REORDERING_TASKS
    ):
        question = _SENTENCE_REORDERING_INSTRUCTIONS + "\n" + scrambled
        metadata = {
            "source_dataset": "sentence_reordering",
            "source_index": index,
            "alt_answers": alt_answers,
        }
        tasks.append((question, answer, metadata))
    return tasks


# ──────────────────────────────────────────────────
# Challenge registration
# ──────────────────────────────────────────────────

# Challenge-specific scoring functions, keyed by challenge name.
# Each function takes (response, expected_answer, metadata) and returns a score in [0, 1].
_CHALLENGE_SCORERS: dict[str, Callable[[str, str, dict[str, Any]], float]] = {
    "ab": _score_ab,
    "advanced_geometry": _score_advanced_geometry,
    "knights_knaves": _score_knights_knaves,
    "futoshiki": _score_futoshiki,
    "jugs": _score_jugs,
    "sentence_reordering": _score_sentence_reordering,
}

# Challenge-specific task builders, keyed by challenge name.
_CHALLENGE_BUILDERS: dict[str, Callable[[], list[tuple[str, str, dict[str, Any]]]]] = {
    "ab": _build_ab_tasks,
    "advanced_geometry": _build_advanced_geometry_tasks,
    "knights_knaves": _build_knights_knaves_tasks,
    "futoshiki": _build_futoshiki_tasks,
    "jugs": _build_jugs_tasks,
    "sentence_reordering": _build_sentence_reordering_tasks,
}


class Settings(BaseModel):
    challenges: list[str] = Field(
        default=[
            "ab",
            "advanced_geometry",
            "knights_knaves",
            "futoshiki",
            "jugs",
            "sentence_reordering",
        ],
        description="List of challenge names to evaluate.",
    )

    system_prompt: str = Field(
        default=(
            "In your output, return only the answer in the format requested below "
            "and nothing else."
        ),
        description="System prompt prepended to all challenge questions.",
    )

    print_responses: bool = Field(
        default=False,
        description="Whether to print prompt/response pairs when scoring challenges.",
    )


class CapabilityEval(Scorer):
    """
    Evaluates model capabilities using structured reasoning challenges.

    Each challenge presents questions with known correct answers.
    The score is the harmonic mean of per-challenge accuracies.
    """

    settings: Settings

    @property
    def score_name(self) -> str:
        return "Capability"

    def init(self, ctx: Context) -> None:
        print()
        print("Loading capability evaluation challenges...")

        self._prompts: list[Prompt] = []
        self._answers: list[str] = []
        self._metadata: list[dict[str, Any]] = []

        system_prompt = self.settings.system_prompt

        for challenge_name in self.settings.challenges:
            if challenge_name not in _CHALLENGE_BUILDERS:
                raise ValueError(
                    f"Unknown challenge: {challenge_name}. "
                    f"Available: {', '.join(sorted(_CHALLENGE_BUILDERS))}"
                )

            tasks = _CHALLENGE_BUILDERS[challenge_name]()
            print(f"* [bold]{challenge_name}[/]: {len(tasks)} tasks loaded")

            for question, answer, metadata in tasks:
                self._prompts.append(Prompt(system=system_prompt, user=question))
                self._answers.append(answer)
                self._metadata.append(metadata)

        print(f"* [bold]{len(self._prompts)}[/] total tasks")

    def get_score(self, ctx: Context) -> Score:
        responses = ctx.get_responses(self._prompts)

        # Collect per-task scores grouped by challenge type.
        challenge_scores: dict[str, list[float]] = {}
        for prompt, response, answer, metadata in zip(
            self._prompts, responses, self._answers, self._metadata
        ):
            scorer_function = _CHALLENGE_SCORERS[metadata["source_dataset"]]
            task_score = scorer_function(response, answer, metadata)
            dataset = metadata["source_dataset"]
            challenge_scores.setdefault(dataset, []).append(task_score)

            if self.settings.print_responses:
                print()
                print(f"[bold]System prompt:[/] {prompt.system}")
                print(f"[bold]Prompt:[/] {prompt.user}")
                if not response.strip():
                    response = "[italic]\\[empty][/]"
                color = "green" if task_score >= 1.0 else "red"
                print(f"[bold]Response:[/] [{color}]{response}[/]")
                print(f"[bold]Expected:[/] {answer}")
                print(f"[bold]Score:[/] {task_score:g}")

        if self.settings.print_responses:
            print()

        # Normalize each challenge type to 0..1 (mean accuracy per challenge).
        challenge_means: dict[str, float] = {}
        for name, scores in challenge_scores.items():
            challenge_means[name] = sum(scores) / len(scores)

        # Harmonic mean across challenge types (0 if any challenge scores 0).
        n = len(challenge_means)
        if n == 0 or any(v == 0.0 for v in challenge_means.values()):
            hmean = 0.0
        else:
            hmean = n / sum(1.0 / v for v in challenge_means.values())

        # Build per-challenge display.
        parts = [
            f"{name}:{challenge_means[name]:.0%}"
            for name in self.settings.challenges
            if name in challenge_means
        ]
        detail = " ".join(parts)

        return Score(
            value=hmean,
            cli_display=f"{hmean:.1%} hmean [{detail}]",
            md_display=f"{hmean:.1%} hmean [{detail}]",
        )
