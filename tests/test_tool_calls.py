import unittest

from heretic.model import detect_tool_call_syntax
from heretic.scorers.llm_judge import get_submitted_answer
from heretic.scorers.quality_canary import get_call, is_correct, is_passed

# The call syntax of Qwen3.5's chat template.
XML_CALL = (
    "\n\n<tool_call>\n<function=submit_answer>\n<parameter=gas_formula>\nCO2\n</parameter>\n"
    '<parameter=steps>\n["a", "b"]\n</parameter>\n</function>\n</tool_call>'
)
JSON_CALL = '<tool_call>\n{"name": "report_unanswerable", "arguments": {"missing_information": "x"}}\n</tool_call>'
XML_ANSWER = "<tool_call>\n<function=submit_answer>\n<parameter=answer>\nOnce upon\na time.\n</parameter>\n</function>\n</tool_call>"


class Template:
    """A tokenizer whose chat template renders messages with the given function."""

    def __init__(self, render):
        self.render = render

    def apply_chat_template(self, messages, tokenize):
        return self.render(messages)


def fail(messages):
    raise ValueError("no tool calls")


class TestToolCalls(unittest.TestCase):
    def test_detection(self):
        def xml(messages):
            call = messages[-1]["tool_calls"][0]["function"]
            return f"<function={call['name']}>\n<parameter=value>\n1\n</parameter>\n</function>"

        self.assertEqual(detect_tool_call_syntax(Template(xml)), "xml")  # ty:ignore[invalid-argument-type]
        self.assertEqual(detect_tool_call_syntax(Template(str)), "json")  # ty:ignore[invalid-argument-type]
        self.assertEqual(detect_tool_call_syntax(Template(fail)), "json")  # ty:ignore[invalid-argument-type]

    def test_xml_call(self):
        call = ("submit_answer", {"gas_formula": "CO2", "steps": ["a", "b"]})
        self.assertEqual(get_call(XML_CALL, "xml"), call)
        self.assertIsNone(get_call(XML_CALL))
        self.assertEqual(get_call("The gas is CO2." + XML_CALL, "xml"), call)
        self.assertIsNone(get_call(XML_CALL + "\nThe gas is CO2.", "xml"))

    def test_json_call(self):
        call = ("report_unanswerable", {"missing_information": "x"})
        self.assertEqual(get_call(JSON_CALL), call)
        self.assertEqual(get_call("Something is missing.\n" + JSON_CALL), call)
        self.assertIsNone(get_call(JSON_CALL + "\nSomething is missing."))
        self.assertIsNone(get_call(JSON_CALL, "xml"))

    def test_values_hold_only_the_answer(self):
        number = {
            "kind": "num",
            "key": 1.61,
            "tolerance": {"absolute": 0.01, "relative": 0},
        }
        self.assertTrue(is_correct(number, " 1.61 "))
        self.assertTrue(is_correct(number, 1.61))
        self.assertFalse(is_correct(number, "1.61 atm"))
        self.assertFalse(is_correct(number, "about 1.61"))
        self.assertTrue(is_correct({"kind": "int", "key": 1234}, "1,234"))
        moment = {"kind": "datetime", "key": "2026-03-22 09:15"}
        self.assertTrue(is_correct(moment, "2026-03-22 09:15"))
        self.assertFalse(is_correct(moment, "Arrives 2026-03-22 09:15"))
        protein = {"kind": "protein", "key": "MKV"}
        self.assertTrue(is_correct(protein, "mkv"))
        self.assertFalse(is_correct(protein, "MKV*"))
        self.assertFalse(is_correct(protein, "M-K-V"))

    def test_no_extra_arguments(self):
        question = {
            "expect": "submit",
            "fields": [{"name": "dose", "check": {"kind": "enum", "key": "10 mg"}}],
        }
        self.assertTrue(is_passed(question, ("submit_answer", {"dose": "10 mg"})))
        self.assertFalse(
            is_passed(question, ("submit_answer", {"dose": "10 mg", "why": "x"}))
        )

    def test_xml_answer(self):
        self.assertEqual(get_submitted_answer(XML_ANSWER, "xml"), "Once upon\na time.")
        self.assertIsNone(get_submitted_answer(XML_ANSWER))
        self.assertEqual(
            get_submitted_answer("Here it is.\n" + XML_ANSWER, "xml"),
            "Once upon\na time.",
        )
        self.assertIsNone(
            get_submitted_answer(XML_ANSWER + "\nI hope you like it.", "xml")
        )
        extra = XML_ANSWER.replace(
            "</function>", "<parameter=note>\nx\n</parameter>\n</function>"
        )
        self.assertIsNone(get_submitted_answer(extra, "xml"))

    def test_xml_answer_cut_off(self):
        text = "<tool_call>\n<function=submit_answer>\n<parameter=answer>\nOnce upon a"
        self.assertEqual(get_submitted_answer(text, "xml"), "Once upon a")

    def test_json_answer(self):
        text = '<tool_call>\n{"name": "submit_answer", "arguments": {"answer": "B"}}\n</tool_call>'
        self.assertEqual(get_submitted_answer(text), "B")
        self.assertEqual(get_submitted_answer("I pick B.\n" + text), "B")
        self.assertIsNone(
            get_submitted_answer(text.replace('"B"}', '"B", "why": "x"}'))
        )
        cut_off = '<tool_call>\n{"name": "submit_answer", "arguments": {"answer": "Once upon a'
        self.assertEqual(get_submitted_answer(cut_off), "Once upon a")


if __name__ == "__main__":
    unittest.main()
