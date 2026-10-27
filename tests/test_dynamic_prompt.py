import unittest
from src.extraction import prompts


class TestDynamicPrompts(unittest.TestCase):

    def test_prescription_prompt_is_dynamic(self):
        prompt = prompts.PRESCRIPTION_EXTRACTION_PROMPT
        self.assertNotIn("Return exactly this structure", prompt)
        lower = prompt.lower()
        self.assertNotIn("investigations", lower)
        self.assertIn("document_type", prompt)

    def test_lab_report_prompt_is_dynamic(self):
        prompt = prompts.LAB_REPORT_EXTRACTION_PROMPT
        self.assertNotIn("Return exactly this structure", prompt)
        self.assertIn("document_type", prompt)

    def test_build_medical_extraction_prompt_for_unseen_document(self):
        prompt = prompts.build_medical_extraction_prompt("Medical_Form")
        self.assertIn("Medical_Form", prompt)
        self.assertIn("document_type", prompt)
        self.assertNotIn("predefined schema", prompt.lower())


if __name__ == "__main__":
    unittest.main()

