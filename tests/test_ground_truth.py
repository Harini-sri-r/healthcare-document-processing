import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api import app
from src.db.database import Base, get_db
from src.db.models import Document


PUBLIC_API_PATHS = {
    "/documents/process",
    "/documents/{document_id}",
    "/patients/{patient_id}/timeline",
    "/documents/search/",
    "/analytics/documents",
}


@pytest.fixture()
def client(tmp_path):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    TestingSession = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(engine)
    original = tmp_path / "source.png"
    original.write_bytes(b"local test image placeholder")
    db = TestingSession()
    db.add_all([
        Document(id="doc-draft", document_type="Prescription", original_filename="draft.png",
                 original_document_path=str(original), extraction_status="SUCCESS"),
        Document(id="doc-verified", document_type="Lab_Report", original_filename="verified.png",
                 original_document_path=str(original), extraction_status="SUCCESS"),
    ])
    db.commit(); db.close()

    def override_db():
        session = TestingSession()
        try: yield session
        finally: session.close()
    app.dependency_overrides[get_db] = override_db
    with TestClient(app) as test_client: yield test_client
    app.dependency_overrides.pop(get_db, None)
    Base.metadata.drop_all(engine); engine.dispose()


def test_create_edit_persist_dynamic_data_and_revision_history(client):
    dynamic = {"patient": {"custom field": "A", "measurements": [{"name": "x", "value": 4}]}}
    first = client.put("/ground-truth/documents/doc-draft", json={
        "data": dynamic, "field_statuses": {"patient.custom field": "Verified", "patient.measurements[0].value": "Needs Review"}})
    assert first.status_code == 200, first.text
    assert first.json()["status"] == "Draft"
    edited = {"patient": {"custom field": "B", "measurements": [{"name": "x", "value": 5}]}}
    second = client.put("/ground-truth/documents/doc-draft", json={"data": edited, "field_statuses": {"patient.custom field": "Unreadable"}})
    assert second.status_code == 200
    assert len(second.json()["revisions"]) == 2
    loaded = client.get("/ground-truth/documents/doc-draft").json()["annotation"]
    assert loaded["data"] == edited
    assert loaded["field_statuses"]["patient.custom field"] == "Unreadable"
    assert len(loaded["revisions"]) == 2


def test_json_validation_and_explicit_verification(client):
    assert client.put("/ground-truth/documents/doc-draft", json={"data": "must be object or array"}).status_code == 422
    invalid_number = client.put("/ground-truth/documents/doc-draft", content='{"data":{"x":NaN}}', headers={"Content-Type": "application/json"})
    assert invalid_number.status_code == 422
    assert client.put("/ground-truth/documents/doc-draft", json={"data": {}, "field_statuses": {"x": "Magic"}}).status_code == 422
    client.put("/ground-truth/documents/doc-draft", json={"data": {"result": [1, {"ok": True}]}})
    assert client.post("/ground-truth/documents/doc-draft/workflow", json={"status": "Verified"}).status_code == 400
    assert client.put("/ground-truth/documents/doc-draft", json={"data": {}, "status": "Verified"}).status_code == 422
    verified = client.post("/ground-truth/documents/doc-draft/workflow", json={"status": "Verified", "confirmed": True})
    assert verified.status_code == 200
    assert verified.json()["status"] == "Verified"
    edited = client.put("/ground-truth/documents/doc-draft", json={"data": {"changed": True}})
    assert edited.json()["status"] == "Draft"
    assert client.get("/ground-truth/documents/doc-draft").json()["annotation"]["status"] == "Draft"


def test_verified_export_excludes_drafts_and_csv_flattens(client):
    client.put("/ground-truth/documents/doc-draft", json={"data": {"nested": [{"value": "draft"}]}})
    client.put("/ground-truth/documents/doc-verified", json={
        "data": {"nested": [{"value": "verified"}]}, "field_statuses": {"nested[0].value": "Verified"}})
    client.post("/ground-truth/documents/doc-verified/workflow", json={"status": "Verified", "confirmed": True})
    j = client.get("/ground-truth/export.json")
    assert j.status_code == 200
    exported = j.json()
    assert [row["document_id"] for row in exported] == ["doc-verified"]
    assert exported[0]["data"]["nested"][0]["value"] == "verified"
    csv_text = client.get("/ground-truth/export.csv").content.decode("utf-8-sig")
    assert "doc-verified" in csv_text and "nested[0].value" in csv_text
    assert "doc-draft" not in csv_text and "draft" not in csv_text


def test_original_source_route_and_delete_preserve_document(client):
    assert client.get("/ground-truth/documents/doc-draft/source").status_code == 200


def test_openapi_documents_only_the_five_public_api_endpoints(client):
    schema_response = client.get("/openapi.json")
    assert schema_response.status_code == 200
    assert set(schema_response.json()["paths"]) == PUBLIC_API_PATHS

    docs_response = client.get("/docs")
    assert docs_response.status_code == 200
    assert "url: '/openapi.json'" in docs_response.text


def test_hidden_ground_truth_route_remains_functional_and_undocumented(client):
    response = client.put("/ground-truth/documents/doc-draft", json={"data": {"still": "works"}})
    assert response.status_code == 200
    assert response.json()["data"] == {"still": "works"}
    assert "/ground-truth/documents/{document_id}" not in client.get("/openapi.json").json()["paths"]
    client.put("/ground-truth/documents/doc-draft", json={"data": {"x": 1}})
    assert client.delete("/ground-truth/documents/doc-draft").json() == {"deleted": True}
    assert client.get("/documents/doc-draft").status_code == 200
    assert client.get("/ground-truth/documents/doc-draft/source").status_code == 200
