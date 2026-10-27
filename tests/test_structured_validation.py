from src.extraction.json_validator import JSONValidator, clean_and_parse_json


def test_different_dynamic_document_structures():
    validator = JSONValidator()

    # Test A: Lab Report with dynamic results array
    test_a = {
        "document_type": "Lab_Report",
        "results": [
            {"assay_name": "Hemoglobin", "value": "13.5", "units": "g/dL"},
            {"assay_name": "Platelets", "value": "250,000", "units": "/uL"}
        ]
    }
    res_a = validator.validate({"success": True, "data": test_a}, "Lab_Report")
    assert res_a["valid"]

    # Test B: Prescription with dynamic medications
    test_b = {
        "document_type": "Prescription",
        "medications": [
            {"drug": "Amoxicillin", "dose": "500mg", "freq": "TID"}
        ]
    }
    res_b = validator.validate({"success": True, "data": test_b}, "Prescription")
    assert res_b["valid"]

    # Test C: Discharge Summary with dynamic sections
    test_c = {
        "document_type": "Discharge_Summary",
        "admission": {"date": "2026-01-01"},
        "follow_up": ["Cardiology clinic in 2 weeks"]
    }
    res_c = validator.validate({"success": True, "data": test_c}, "Discharge_Summary")
    assert res_c["valid"]

    # Test D: Medical Form with dynamic form sections
    test_d = {
        "document_type": "Medical_Form",
        "form_sections": [
            {"heading": "Insurance", "policy_number": "POL-9988"}
        ]
    }
    res_d = validator.validate({"success": True, "data": test_d}, "Medical_Form")
    assert res_d["valid"]


def test_nested_objects_and_arrays():
    validator = JSONValidator()
    dynamic_payload = {
        "document_type": "Other_Healthcare_Document",
        "header": {
            "facility": {"name": "General Hospital", "city": "Cityville"},
            "ref_code": 12345
        },
        "sections": [
            {"title": "Section 1", "items": ["Item A", "Item B"]},
            {"title": "Section 2", "details": {"sub_field": True}}
        ]
    }
    res = validator.validate({"success": True, "data": dynamic_payload})
    assert res["valid"]
    assert not res["requires_review"]


def test_null_and_unclear_values():
    validator = JSONValidator()
    payload = {
        "document_type": "Lab_Report",
        "patient_name": "John Doe",
        "test_val": "[UNCLEAR]",
        "optional_field": None,
        "source": {
            "uncertain_fields": ["test_val"]
        }
    }
    res = validator.validate({"success": True, "data": payload}, "Lab_Report")
    assert res["valid"]
    assert res["requires_review"]


def test_document_type_mismatch():
    validator = JSONValidator()
    payload = {"document_type": "Prescription", "medications": []}
    res = validator.validate({"success": True, "data": payload}, "Lab_Report")
    assert res["valid"]
    # Check that an informational issue for type mismatch was generated
    assert any(i["type"] == "type_mismatch" for i in res["issues"])


def test_invalid_json_payload():
    validator = JSONValidator()
    res_fail = validator.validate({"success": False, "error": "LLM failed"})
    assert not res_fail["valid"]
    assert res_fail["requires_review"]

    res_none = validator.validate({"success": True, "data": None})
    assert not res_none["valid"]

    res_list = validator.validate({"success": True, "data": ["not", "a", "dict"]})
    assert not res_list["valid"]


def test_truncated_json_is_not_repaired_by_inventing_content():
    parsed, error = clean_and_parse_json('{"patient": {"name": "A"')
    assert parsed is None
    assert error
    parsed, error = clean_and_parse_json('{"valid": 1}')
    assert parsed == {"valid": 1}
    assert error is None
    parsed, error = clean_and_parse_json('{"patient": {"name": "A"}} incomplete')
    assert parsed is None
    assert error


def test_pdf_page_loader_keeps_all_pages(tmp_path):
    import fitz
    from src.preprocessing.image_preprocessing import load_document_pages

    path = tmp_path / "two-pages.pdf"
    pdf = fitz.open()
    pdf.new_page(width=200, height=300)
    pdf.new_page(width=200, height=300)
    pdf.save(path)
    pdf.close()
    assert len(load_document_pages(path)) == 2

