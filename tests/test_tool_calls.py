import unittest

from heretic.model import detect_tool_call_syntax
from heretic.scorers.llm_judge import get_submitted_answer
from heretic.scorers.quality_canary import get_call

# The call syntax of Qwen3.5's chat template.
XML_CALL = (
    "\n\n<tool_call>\n<function=submit_answer>\n<parameter=gas_formula>\nCO2\n</parameter>\n"
    '<parameter=steps>\n["a", "b"]\n</parameter>\n</function>\n</tool_call>'
)
JSON_CALL = '<tool_call>\n{"name": "report_unanswerable", "arguments": {"missing_information": "x"}}\n</tool_call>'


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
        self.assertEqual(
            get_call(XML_CALL, bare=True, syntax="xml"),
            ("submit_answer", {"gas_formula": "CO2", "steps": ["a", "b"]}),
        )
        self.assertIsNone(get_call(XML_CALL, bare=True))

    def test_xml_call_after_reasoning(self):
        text = "The gas is CO2." + XML_CALL
        self.assertIsNone(get_call(text, bare=True, syntax="xml"))
        self.assertEqual(get_call(text, bare=False, syntax="xml")[0], "submit_answer")  # ty:ignore[not-subscriptable]

    def test_json_call(self):
        self.assertEqual(
            get_call(JSON_CALL, bare=True),
            ("report_unanswerable", {"missing_information": "x"}),
        )
        self.assertIsNone(get_call(JSON_CALL, bare=True, syntax="xml"))

    def test_xml_answer(self):
        text = "<tool_call>\n<function=submit_answer>\n<parameter=answer>\nOnce upon\na time.\n</parameter>\n</function>\n</tool_call>"
        self.assertEqual(get_submitted_answer(text, "xml"), "Once upon\na time.")
        self.assertIsNone(get_submitted_answer(text))

    def test_xml_answer_cut_off(self):
        text = "<tool_call>\n<function=submit_answer>\n<parameter=answer>\nOnce upon a"
        self.assertEqual(get_submitted_answer(text, "xml"), "Once upon a")

    def test_xml_answer_with_commentary(self):
        text = "<tool_call>\n<function=submit_answer>\n<parameter=answer>\nB\n</parameter>\n</function>\n</tool_call>\nI chose B."
        self.assertIsNone(get_submitted_answer(text, "xml"))


if __name__ == "__main__":
    unittest.main()
