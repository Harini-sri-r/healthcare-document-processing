from fastapi.testclient import TestClient
import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.api import app, get_db
from src.db.database import Base, engine, SessionLocal

client = TestClient(app)

@pytest.fixture(scope="module")
def setup_db():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)

def override_get_db():
    try:
        db = SessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

def test_get_analytics(setup_db):
    response = client.get("/analytics/documents")
    assert response.status_code == 200
    data = response.json()
    assert "total_documents" in data
    assert "documents_by_type" in data

def test_get_review_pending(setup_db):
    response = client.get("/documents/review/pending")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_search_documents(setup_db):
    response = client.get("/documents/search/?keyword=test")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_get_nonexistent_document(setup_db):
    response = client.get("/documents/fake-id")
    assert response.status_code == 404

def test_get_patient_timeline(setup_db):
    response = client.get("/patients/patient-123/timeline")
    assert response.status_code == 200
    data = response.json()
    assert "patient_id" in data
    assert "timeline" in data
    assert isinstance(data["timeline"], list)


def test_pdf_extension_is_supported_by_upload_route():
    from app.api import ALLOWED_EXTENSIONS
    assert ".pdf" in ALLOWED_EXTENSIONS
