import unittest

from heretic.scorers.llm_judge import get_submitted_answer
from heretic.scorers.quality_canary import get_call

# The call syntax of Qwen3.5's chat template.
XML_CALL = (
    "\n\n<tool_call>\n<function=submit_answer>\n<parameter=gas_formula>\nCO2\n</parameter>\n"
    '<parameter=steps>\n["a", "b"]\n</parameter>\n</function>\n</tool_call>'
)


class TestToolCalls(unittest.TestCase):
    def test_xml_call(self):
        self.assertEqual(
            get_call(XML_CALL, bare=True),
            ("submit_answer", {"gas_formula": "CO2", "steps": ["a", "b"]}),
        )

    def test_xml_call_after_reasoning(self):
        text = "The gas is CO2." + XML_CALL
        self.assertIsNone(get_call(text, bare=True))
        self.assertEqual(get_call(text, bare=False)[0], "submit_answer")  # ty:ignore[not-subscriptable]

    def test_json_call(self):
        text = '<tool_call>\n{"name": "report_unanswerable", "arguments": {"missing_information": "x"}}\n</tool_call>'
        self.assertEqual(
            get_call(text, bare=True),
            ("report_unanswerable", {"missing_information": "x"}),
        )

    def test_xml_answer(self):
        text = "<tool_call>\n<function=submit_answer>\n<parameter=answer>\nOnce upon\na time.\n</parameter>\n</function>\n</tool_call>"
        self.assertEqual(get_submitted_answer(text), "Once upon\na time.")

    def test_xml_answer_cut_off(self):
        text = "<tool_call>\n<function=submit_answer>\n<parameter=answer>\nOnce upon a"
        self.assertEqual(get_submitted_answer(text), "Once upon a")

    def test_xml_answer_with_commentary(self):
        text = "<tool_call>\n<function=submit_answer>\n<parameter=answer>\nB\n</parameter>\n</function>\n</tool_call>\nI chose B."
        self.assertIsNone(get_submitted_answer(text))


if __name__ == "__main__":
    unittest.main()
