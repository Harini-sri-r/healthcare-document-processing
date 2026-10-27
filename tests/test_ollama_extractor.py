"""Unit tests for OllamaExtractor — runs without Ollama server."""
import base64
import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image

from src.config import settings
from src.extraction.ollama_extractor import OllamaExtractor
from src.extraction.prompts import build_medical_extraction_prompt
from src.extraction.json_validator import JSONValidator


class OllamaExtractorInitTests(unittest.TestCase):
    def test_default_initialization(self):
        extractor = OllamaExtractor()
        self.assertEqual(extractor.base_url, settings.OLLAMA_BASE_URL.rstrip("/"))
        self.assertEqual(extractor.model, settings.OLLAMA_MODEL)
        self.assertEqual(extractor.timeout, settings.OLLAMA_TIMEOUT)
        self.assertEqual(extractor.max_retries, settings.QWEN_MAX_RETRIES)

    def test_custom_initialization(self):
        extractor = OllamaExtractor(
            base_url="http://192.168.1.10:11434",
            model="custom-model:7b",
            timeout=60,
            max_retries=5,
        )
        self.assertEqual(extractor.base_url, "http://192.168.1.10:11434")
        self.assertEqual(extractor.model, "custom-model:7b")
        self.assertEqual(extractor.timeout, 60)
        self.assertEqual(extractor.max_retries, 5)


class ImageEncodingTests(unittest.TestCase):
    def test_encode_image_returns_base64_string(self):
        img = Image.new("RGB", (32, 32), color="red")
        encoded = OllamaExtractor._encode_image(img)
        self.assertIsInstance(encoded, str)
        self.assertGreater(len(encoded), 0)

        # Verify it's valid base64
        decoded = base64.b64decode(encoded)
        self.assertGreater(len(decoded), 0)


class CheckConnectionTests(unittest.TestCase):
    @patch("src.extraction.ollama_extractor.requests.get")
    def test_connection_success_model_found(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "models": [{"name": "qwen2.5vl:3b"}, {"name": "llama3:latest"}]
        }
        mock_get.return_value = mock_resp

        extractor = OllamaExtractor()
        result = extractor.check_connection()
        self.assertTrue(result["reachable"])
        self.assertTrue(result["model_available"])
        self.assertIsNone(result["error"])

    @patch("src.extraction.ollama_extractor.requests.get")
    def test_connection_success_model_not_found(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "models": [{"name": "llama3:latest"}]
        }
        mock_get.return_value = mock_resp

        extractor = OllamaExtractor()
        result = extractor.check_connection()
        self.assertTrue(result["reachable"])
        self.assertFalse(result["model_available"])
        self.assertIn("not found", result["error"])

    @patch("src.extraction.ollama_extractor.requests.get")
    def test_connection_refused(self, mock_get):
        import requests as req
        mock_get.side_effect = req.ConnectionError("Connection refused")

        extractor = OllamaExtractor()
        result = extractor.check_connection()
        self.assertFalse(result["reachable"])
        self.assertFalse(result["model_available"])
        self.assertIn("Cannot connect", result["error"])


class GenerateRawTextTests(unittest.TestCase):
    @patch("src.extraction.ollama_extractor.requests.post")
    def test_successful_generation(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "message": {"content": '{"test_name": "Hemoglobin", "value": "14.2"}'}
        }
        mock_post.return_value = mock_resp

        extractor = OllamaExtractor()
        img = Image.new("RGB", (32, 32))
        result = extractor.generate_raw_text(img, "Extract data")

        self.assertTrue(result["success"])
        self.assertIn("Hemoglobin", result["text"])
        self.assertIsNone(result["error"])

        payload = mock_post.call_args.kwargs["json"]
        self.assertEqual(payload["model"], "qwen2.5vl:3b")
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["messages"][0]["role"], "user")
        self.assertEqual(payload["messages"][0]["content"], "Extract data")
        self.assertEqual(payload["options"]["num_ctx"], settings.OLLAMA_NUM_CTX)
        self.assertEqual(payload["options"]["num_predict"], 1600)
        self.assertEqual(payload["options"]["num_predict"], settings.OLLAMA_NUM_PREDICT)
        self.assertEqual(payload["options"]["num_ctx"], 8192)
        encoded_image = payload["messages"][0]["images"][0]
        self.assertIsInstance(encoded_image, str)
        self.assertTrue(base64.b64decode(encoded_image).startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertEqual(payload["format"], "json")

    def test_document_specific_prompt_requires_complete_valid_json(self):
        prompt = build_medical_extraction_prompt().lower()

        self.assertIn('"document_type"', prompt)
        self.assertIn("return valid json only", prompt)
        self.assertIn("meaningful field names", prompt)
        self.assertIn("invent", prompt)


    @patch("src.extraction.ollama_extractor.requests.post")
    def test_http_error_includes_ollama_response_body(self, mock_post):
        import requests as req

        response = MagicMock()
        response.status_code = 400
        response.text = '{"error":"invalid request payload"}'
        response.raise_for_status.side_effect = req.HTTPError(
            "400 Client Error: Bad Request", response=response
        )
        mock_post.return_value = response

        result = OllamaExtractor().generate_raw_text(Image.new("RGB", (32, 32)), "Extract data")

        self.assertFalse(result["success"])
        self.assertIn("invalid request payload", result["error"])

    @patch("src.extraction.ollama_extractor.requests.post")
    def test_connection_error(self, mock_post):
        import requests as req
        mock_post.side_effect = req.ConnectionError("Connection refused")

        extractor = OllamaExtractor()
        img = Image.new("RGB", (32, 32))
        result = extractor.generate_raw_text(img, "Extract data")

        self.assertFalse(result["success"])
        self.assertIsNone(result["text"])
        self.assertIn("Cannot connect", result["error"])

    @patch("src.extraction.ollama_extractor.requests.post")
    def test_timeout_error(self, mock_post):
        import requests as req
        mock_post.side_effect = req.Timeout("Request timed out")

        extractor = OllamaExtractor()
        img = Image.new("RGB", (32, 32))
        result = extractor.generate_raw_text(img, "Extract data")

        self.assertFalse(result["success"])
        self.assertIn("timed out", result["error"])

    @patch("src.extraction.ollama_extractor.requests.post")
    def test_empty_response(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"message": {"content": ""}}
        mock_post.return_value = mock_resp

        extractor = OllamaExtractor()
        img = Image.new("RGB", (32, 32))
        result = extractor.generate_raw_text(img, "Extract data")

        self.assertFalse(result["success"])
        self.assertIn("empty response", result["error"])


class ExtractTests(unittest.TestCase):
    @patch("src.extraction.ollama_extractor.requests.post")
    def test_successful_extraction(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "message": {
                "content": json.dumps({
                    "patient_name": "Test Patient",
                    "test_name": "Hemoglobin",
                    "result": "14.2",
                    "unit": "g/dL",
                    "reference_range": "12.0 - 15.0",
                })
            }
        }
        mock_post.return_value = mock_resp

        extractor = OllamaExtractor()
        img = Image.new("RGB", (32, 32))
        result = extractor.extract(img)

        self.assertTrue(result["success"])
        self.assertEqual(result["attempts"], 1)
        self.assertIsNotNone(result["data"])
        self.assertEqual(result["data"]["test_name"], "Hemoglobin")
        self.assertEqual(result["data"]["unit"], "g/dL")
        self.assertIsNone(result["error"])

    @patch("src.extraction.ollama_extractor.requests.post")
    def test_extraction_with_markdown_fences(self, mock_post):
        """Ensure JSON wrapped in ```json...``` fences is parsed correctly."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "message": {
                "content": '```json\n{"test": "WBC", "value": "8.5", "unit": "x10^3/uL"}\n```'
            }
        }
        mock_post.return_value = mock_resp

        extractor = OllamaExtractor()
        img = Image.new("RGB", (32, 32))
        result = extractor.extract(img)

        self.assertTrue(result["success"])
        self.assertEqual(result["data"]["test"], "WBC")

    @patch("src.extraction.ollama_extractor.requests.post")
    def test_extraction_retry_on_malformed_json(self, mock_post):
        """First response is malformed, second is valid — extraction should succeed on attempt 2."""
        bad_resp = MagicMock()
        bad_resp.status_code = 200
        bad_resp.json.return_value = {
            "message": {"content": "This is not JSON at all."}
        }

        good_resp = MagicMock()
        good_resp.status_code = 200
        good_resp.json.return_value = {
            "message": {"content": '{"test": "Glucose", "value": "95"}'}
        }

        mock_post.side_effect = [bad_resp, good_resp]

        extractor = OllamaExtractor(max_retries=2)
        img = Image.new("RGB", (32, 32))
        result = extractor.extract(img)

        self.assertTrue(result["success"])
        self.assertEqual(result["attempts"], 2)
        self.assertEqual(result["data"]["test"], "Glucose")
        self.assertEqual(len(result["history"]), 2)
        self.assertFalse(result["history"][0]["valid_json"])
        self.assertTrue(result["history"][1]["valid_json"])

    @patch("src.extraction.ollama_extractor.requests.post")
    def test_deterministic_repair_of_bare_strings(self, mock_post):
        raw = '{"Sample Collected At": {"Clinic", "Bengaluru"}, "Platelets": {"count is at upper limit"}}'
        response = MagicMock()
        response.json.return_value = {"message": {"content": raw}}
        mock_post.return_value = response

        result = OllamaExtractor(max_retries=0).extract(Image.new("RGB", (32, 32)))
        self.assertTrue(result["success"])
        self.assertEqual(result["data"]["Sample Collected At"], {"items": ["Clinic", "Bengaluru"]})
        self.assertEqual(result["data"]["Platelets"], {"text": "count is at upper limit"})
        self.assertEqual(result["data"]["Platelets"]["text"], "count is at upper limit")
        self.assertNotIn("raw_output", result["history"][0])
        self.assertEqual(mock_post.call_count, 1)

    @patch("src.extraction.ollama_extractor.requests.post")
    def test_repair_preserves_valid_fields_and_does_not_drop_fields(self, mock_post):
        raw = '{"Peripheral Blood Smear": {"RBCs": "normocytic", "No haemolysis seen."}, "Platelets": {"count is at upper limit"}, "other": 4}'
        response = MagicMock()
        response.json.return_value = {"message": {"content": raw}}
        mock_post.return_value = response

        result = OllamaExtractor(max_retries=0).extract(Image.new("RGB", (32, 32)))
        self.assertTrue(result["success"])
        self.assertEqual(result["data"], {
            "Peripheral Blood Smear": {"RBCs": "normocytic", "text": "No haemolysis seen."},
            "Platelets": {"text": "count is at upper limit"}, "other": 4,
        })
        self.assertEqual(mock_post.call_count, 1)

    @patch("src.extraction.ollama_extractor.requests.post")
    def test_truncated_json_is_not_guessed_or_retained(self, mock_post):
        raw = '{"test": "value", "unfinished": '
        response = MagicMock()
        response.json.return_value = {"message": {"content": raw}}
        mock_post.return_value = response

        result = OllamaExtractor(max_retries=0).extract(Image.new("RGB", (32, 32)))
        self.assertFalse(result["success"])
        self.assertNotIn("raw_output", result["history"][0])
        self.assertFalse(result["history"][0]["valid_json"])

    def test_clean_valid_json_passes_unchanged(self):
        from src.extraction.json_validator import clean_and_parse_json

        raw = '{"valid": {"key": "value"}, "list": [1, 2]}'
        parsed, error = clean_and_parse_json(raw)
        self.assertIsNone(error)
        self.assertEqual(parsed, json.loads(raw))

    @patch("src.extraction.ollama_extractor.requests.post")
    def test_malformed_comma_retries_with_parser_feedback_and_preserves_raw(self, mock_post):
        malformed = '{"patient": {"name": "A" "age": 30}, "items": [true, null]}'
        good = '{"patient": {"name": "A", "age": 30}, "items": [true, null]}'
        responses = []
        for content in (malformed, good):
            response = MagicMock()
            response.json.return_value = {"message": {"content": content}}
            responses.append(response)
        mock_post.side_effect = responses
        result = OllamaExtractor(max_retries=1, preserve_raw_output=True).extract(Image.new("RGB", (8, 8)))
        self.assertTrue(result["success"])
        self.assertEqual(result["data"], {"patient": {"name": "A", "age": 30}, "items": [True, None]})
        self.assertEqual(result["raw_output"], good)
        second_prompt = mock_post.call_args_list[1].kwargs["json"]["messages"][0]["content"]
        self.assertIn("Parser feedback", second_prompt)
        self.assertIn("Expecting", second_prompt)

    @patch("src.extraction.ollama_extractor.requests.post")
    def test_failed_parse_retains_last_raw_response_and_truncation_reason(self, mock_post):
        raw = '{"nested": {"result": [1, 2'
        response = MagicMock()
        response.json.return_value = {"message": {"content": raw}}
        mock_post.return_value = response
        result = OllamaExtractor(max_retries=0, preserve_raw_output=True).extract(Image.new("RGB", (8, 8)))
        self.assertFalse(result["success"])
        self.assertEqual(result["raw_output"], raw)
        self.assertIn("truncated", result["error"].lower())

    @patch("src.extraction.ollama_extractor.requests.post")
    def test_request_timeout_is_bounded_by_overall_deadline(self, mock_post):
        response = MagicMock()
        response.json.return_value = {"message": {"content": '{"dynamic": [1, {"ok": true}]}'}}
        mock_post.return_value = response
        extractor = OllamaExtractor(timeout=90, overall_timeout=2)
        self.assertTrue(extractor.extract(Image.new("RGB", (8, 8)))["success"])
        self.assertLessEqual(sum(mock_post.call_args.kwargs["timeout"]), 2.1)

    @patch("src.extraction.ollama_extractor.requests.post")
    def test_extraction_timeout_is_reported_per_attempt(self, mock_post):
        import requests as req
        mock_post.side_effect = req.Timeout("read timed out")
        result = OllamaExtractor(timeout=1, overall_timeout=5, max_retries=0,
                                 preserve_raw_output=True).extract(Image.new("RGB", (8, 8)))
        self.assertFalse(result["success"])
        self.assertEqual(result["attempts"], 1)
        self.assertIsNone(result["raw_output"])
        self.assertIn("timed out", result["error"].lower())
        self.assertIn("duration_ms", result["history"][0])

    @patch("src.extraction.ollama_extractor.requests.post")
    def test_malformed_json_is_logged_without_retaining_content(self, mock_post):
        malformed_response = '{\n  "patient_name" "Jane Doe",\n  "result": "12"\n}'
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"message": {"content": malformed_response}}
        mock_post.return_value = mock_resp

        extractor = OllamaExtractor(max_retries=0)
        with self.assertLogs("ollama_extractor", level="WARNING") as logs:
            result = extractor.extract(Image.new("RGB", (32, 32)))

        self.assertFalse(result["success"])
        self.assertNotIn("raw_output", result["history"][0])
        self.assertIn("JSON parse error", result["history"][0]["error"])
        self.assertNotIn(malformed_response, "\n".join(logs.output))

    @patch("src.extraction.ollama_extractor.requests.post")
    def test_extraction_connection_error_stops_retrying(self, mock_post):
        import requests as req
        mock_post.side_effect = req.ConnectionError("Refused")

        extractor = OllamaExtractor(max_retries=3)
        img = Image.new("RGB", (32, 32))
        result = extractor.extract(img)

        self.assertFalse(result["success"])
        # Should stop after first attempt since server is unreachable
        self.assertEqual(result["attempts"], 1)
        self.assertIn("Cannot connect", result["history"][0]["error"])

    def test_extraction_invalid_image_path(self):
        extractor = OllamaExtractor()
        result = extractor.extract("non_existent_image_xyz.png")

        self.assertFalse(result["success"])
        self.assertEqual(result["attempts"], 0)
        self.assertIn("Failed to load image", result["error"])


class ResultStructureTests(unittest.TestCase):
    """Ensure OllamaExtractor returns the exact same structure as QwenExtractor."""

    REQUIRED_KEYS = {"success", "attempts", "data", "history", "error"}

    @patch("src.extraction.ollama_extractor.requests.post")
    def test_success_result_has_required_keys(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "message": {"content": '{"result": "ok"}'}
        }
        mock_post.return_value = mock_resp

        extractor = OllamaExtractor()
        result = extractor.extract(Image.new("RGB", (32, 32)))
        self.assertEqual(set(result.keys()), self.REQUIRED_KEYS)

    @patch("src.extraction.ollama_extractor.requests.post")
    def test_failure_result_has_required_keys(self, mock_post):
        import requests as req
        mock_post.side_effect = req.ConnectionError("Refused")
        result = OllamaExtractor().extract(Image.new("RGB", (32, 32)))
        self.assertEqual(set(result.keys()), self.REQUIRED_KEYS)


class StructuredSchemaTests(unittest.TestCase):
    REQUIRED_KEYS = {"success", "attempts", "data", "history", "error"}
    def setUp(self):
        self.validator = JSONValidator()

    def validate(self, kind, data):
        return self.validator.validate({"success": True, "data": data}, kind)


    def test_prescription_dynamic_medications_and_uncertainty(self):
        data = {"document_type": "Prescription", "patient": {"name": None, "age": "29 years", "sex": None},
                "doctor": {"name": None, "address": None, "telephone": None},
                "prescription": {"date": None, "medications": [
                    {"name": "Drug A", "strength": "5 mg", "dosage": "1", "quantity": None, "frequency": "BD", "instructions": None},
                    {"name": "Drug B", "strength": None, "dosage": None, "quantity": "10", "frequency": None, "instructions": None}]},
                "additional_information": [], "source": {"uncertain_fields": ["patient.name"]}}
        result = self.validate("Prescription", data)
        self.assertTrue(result["valid"])
        self.assertTrue(result["requires_review"])
        self.assertEqual(len(data["prescription"]["medications"]), 2)
        self.assertEqual(data["prescription"]["medications"][0]["frequency"], "BD")

    def test_lab_dynamic_investigations_preserve_original_values(self):
        data = {"document_type": "Lab_Report", "patient": {"name": None, "age": None, "sex": None},
                "report_metadata": {"reference": None, "registered_at": None, "sample_collected_at": None,
                                    "sample_received_at": None, "reported_at": None,
                                    "sample_collection_location": None, "processing_location": None},
                "investigations": [{"name": "MCV (Mean Corpuscular Volume)", "observed_value": 79.7, "unit": "fL", "reference_interval": "76-96", "additional_information": "Derived from RBC Histogram"},
                                   {"name": "MCH (Mean Corpuscular Hb)", "observed_value": 26.4, "unit": "pg", "reference_interval": "27-32", "additional_information": "Calculated"}],
                "observations": [{"section": "Peripheral Blood Smear", "text": "No immature RBCs seen."}],
                "interpretation": {"impression": "NORMOCYTIC HYPOCHROMIC ANAEMIA.", "suggested": "Clinical correlation"},
                "additional_information": [], "source": {"uncertain_fields": []}}
        result = self.validate("Lab_Report", data)
        self.assertTrue(result["valid"])
        self.assertFalse(result["requires_review"])
        self.assertEqual(data["investigations"][0]["reference_interval"], "76-96")
        self.assertEqual(data["investigations"][0]["unit"], "fL")
        self.assertEqual(data["investigations"][1]["name"], "MCH (Mean Corpuscular Hb)")
        self.assertEqual(data["investigations"][0]["additional_information"], "Derived from RBC Histogram")

    def test_lab_preserves_patient_metadata_regions_and_bottom_sections(self):
        data = {"document_type": "Lab_Report",
                "patient": {"name": "Mrs. VANITHA U", "age": "42 years", "sex": "Female"},
                "report_metadata": {"reference": "REF-123", "registered_at": "03:48 PM",
                                    "sample_collected_at": "3:45 PM", "sample_received_at": None,
                                    "reported_at": "08:40 PM",
                                    "sample_collection_location": "Vijaya Poly Clinic And Diagnostic Centre\nNo.111, ams Layout, Vidyaranya Pura\nBengaluru",
                                    "processing_location": "Metropolis Healthcare Ltd., #76/10, 15th cross, Malleshwaram, Bangalore."},
                "investigations": [],
                "observations": [{"section": "PERIPHERAL BLOOD SMEAR", "text": "RBCs: predominantly normocytic normochromic/normocytic hypochromic. No immature RBCs seen. No evidence of haemolysis seen. WBCs: normal in number; distribution and morphology. Platelets: count is at upper limit of normal range."}],
                "interpretation": {"impression": "NORMOCYTIC HYPOCHROMIC ANAEMIA.", "suggested": "Clinical correlation"},
                "additional_information": [{"label": "Other Label", "value": "Original text"}],
                "source": {"uncertain_fields": ["report_metadata.sample_received_at"]}}
        result = self.validate("Lab_Report", data)
        self.assertTrue(result["valid"])
        self.assertTrue(result["requires_review"])
        self.assertEqual(data["patient"]["name"], "Mrs. VANITHA U")
        self.assertEqual(data["report_metadata"]["registered_at"], "03:48 PM")
        self.assertEqual(data["report_metadata"]["sample_collected_at"], "3:45 PM")
        self.assertEqual(data["report_metadata"]["reported_at"], "08:40 PM")
        self.assertNotEqual(data["report_metadata"]["sample_collection_location"], data["report_metadata"]["processing_location"])
        self.assertEqual(data["observations"][0]["section"], "PERIPHERAL BLOOD SMEAR")
        self.assertEqual(data["interpretation"]["impression"], "NORMOCYTIC HYPOCHROMIC ANAEMIA.")
        self.assertEqual(data["interpretation"]["suggested"], "Clinical correlation")
        self.assertEqual(data["additional_information"][0]["value"], "Original text")

    def test_lab_prompt_enforces_full_reading_and_region_separation(self):
        from src.extraction.prompts import LAB_REPORT_EXTRACTION_PROMPT
        prompt = LAB_REPORT_EXTRACTION_PROMPT
        for phrase in ("top to bottom", "source.uncertain_fields", "document_type", "RETURN VALID JSON ONLY"):
            self.assertIn(phrase, prompt)
        for hardcoded_test in ('"haemoglobin"', '"mcv"', '"mch"', '"glucose"', '"cholesterol"'):
            self.assertNotIn(hardcoded_test, prompt.lower())

    def test_discharge_dynamic_lists_and_invalid_shape(self):
        data = {"document_type": "Discharge_Summary", "patient": {"name": None, "age": None, "sex": None},
                "admission": {"admission_date": None, "discharge_date": None, "department": None},
                "clinical_information": {"diagnoses": ["Written diagnosis"], "procedures": ["Written procedure"],
                                         "medications": ["Written medication"], "observations": ["Other visible note"]},
                "additional_information": [], "source": {"uncertain_fields": []}}
        self.assertTrue(self.validate("Discharge_Summary", data)["valid"])

    def test_prompts_are_document_specific_and_preserve_safety_constraints(self):
        from src.extraction.prompts import EXTRACTION_PROMPTS
        self.assertEqual(set(EXTRACTION_PROMPTS), {"Prescription", "Lab_Report", "Discharge_Summary"})
        for prompt in EXTRACTION_PROMPTS.values():
            self.assertIn("RETURN VALID JSON ONLY", prompt)
            self.assertIn("uncertain_fields", prompt)

    def test_invalid_json_and_truncation_are_not_repaired(self):
        from src.extraction.json_validator import clean_and_parse_json
        parsed, error = clean_and_parse_json('{"a":')
        self.assertIsNone(parsed)
        self.assertIsNotNone(error)

    def test_all_typed_schemas_accept_dynamic_payloads(self):
        for kind in ("Prescription", "Lab_Report", "Discharge_Summary", "Medical_Form"):
            result = self.validate(kind, {"document_type": kind})
            self.assertTrue(result["valid"])

    @patch("src.extraction.ollama_extractor.requests.post")
    def test_failure_result_has_required_keys(self, mock_post):
        import requests as req
        mock_post.side_effect = req.ConnectionError("Refused")

        extractor = OllamaExtractor()
        result = extractor.extract(Image.new("RGB", (32, 32)))
        self.assertEqual(set(result.keys()), self.REQUIRED_KEYS)


if __name__ == "__main__":
    unittest.main()

