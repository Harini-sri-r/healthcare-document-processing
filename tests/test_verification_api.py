import pytest
from PIL import Image
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app import api as api_module
from src.config import settings
from src.db.database import Base, get_db
from src.db.models import Document, DocumentExtraction, DocumentExtractionRaw, DocumentVerificationRun
from src.services.document_service import DocumentService


@pytest.fixture()
def verification_client(tmp_path, monkeypatch):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    TestingSession = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(engine)
    source = tmp_path / "original.png"; Image.new("RGB", (8, 8), "white").save(source)
    session = TestingSession()
    session.add(Document(id="verify-api-doc", document_type="Prescription", original_filename="test.png",
        original_document_path=str(source), extraction_status="SUCCESS", extraction_processing_status="SUCCESS"))
    session.add(DocumentExtraction(id="extract-id", document_id="verify-api-doc", extraction_json={"dose": "12.5 mg"}))
    session.add(DocumentExtractionRaw(document_id="verify-api-doc", raw_text='{"dose":"12.5 mg"}'))
    session.commit(); session.close()
    def override_db():
        db=TestingSession()
        try: yield db
        finally: db.close()
    api_module.app.dependency_overrides[get_db]=override_db
    monkeypatch.setattr(settings, "SECOND_MODEL_VERIFICATION_ENABLED", True)
    class FakeVerifier:
        def __init__(self): pass
        def verify(self, images, raw, parsed):
            assert len(images)==1 and raw=='{"dose":"12.5 mg"}' and parsed=={"dose":"12.5 mg"}
            return {"verification_model":"local-test-model", "overall_status":"DISCREPANCY_DETECTED",
                "availability":"AVAILABLE", "field_reviews":[{"field_path": '$["dose"]',
                "extracted_value":"12.5 mg", "status":"DISCREPANCY_DETECTED", "observed_source_value":"125 mg", "reason":"different"}],
                "review_notes":[], "latency_ms":123.0, "raw_verifier_output":"{}"}
    monkeypatch.setattr(api_module, "SecondModelVerifier", FakeVerifier)
    with TestClient(api_module.app) as client: yield client, TestingSession
    api_module.app.dependency_overrides.pop(get_db, None); Base.metadata.drop_all(engine); engine.dispose()


def test_verification_run_persists_separately_and_exposes_human_resolution(verification_client):
    client, TestingSession=verification_client
    run=client.post('/documents/verify-api-doc/verify')
    assert run.status_code==200, run.text
    run_data=run.json(); assert run_data['overall_status']=='DISCREPANCY_DETECTED'
    assert run_data['latency_ms']==123.0
    invalid=client.put(f"/verification-runs/{run_data['id']}/resolution",json={
        "field_resolutions": {'$["dose"]': {'action':'corrected'}}, "reviewer_note":"Checked source image"})
    assert invalid.status_code==422
    resolution=client.put(f"/verification-runs/{run_data['id']}/resolution",json={
        "field_resolutions": {'$["dose"]': {'action':'corrected','value':'12.5 mg'}}, "reviewer_note":"Checked source image"})
    assert resolution.status_code==200
    db=TestingSession()
    doc=db.query(Document).filter_by(id='verify-api-doc').one()
    assert doc.extraction.extraction_json=={"dose":"12.5 mg"}
    assert doc.review_required is True
    assert doc.verification_runs[0].report_json['field_reviews'][0]['extracted_value']=='12.5 mg'
    assert doc.verification_runs[0].reviewer_resolution['reviewer_note']=='Checked source image'
    db.close()


def test_verification_disabled_by_default_is_not_run(verification_client, monkeypatch):
    client, _=verification_client
    monkeypatch.setattr(settings, 'SECOND_MODEL_VERIFICATION_ENABLED', False)
    response=client.post('/documents/verify-api-doc/verify')
    assert response.status_code==409


def test_normal_processing_persists_qwen_without_automatic_verification_artifacts(tmp_path, monkeypatch):
    engine=create_engine("sqlite://",connect_args={"check_same_thread":False},poolclass=StaticPool)
    TestingSession=sessionmaker(bind=engine,autoflush=False,autocommit=False)
    Base.metadata.create_all(engine)
    source=tmp_path/"input.png"; Image.new("RGB",(9,9),"white").save(source)
    project=tmp_path/"isolated_project"; fake_file=project/"src"/"services"/"document_service.py"
    monkeypatch.setattr("src.services.document_service.__file__",str(fake_file))
    qwen={"dose":"12.5 mg","nested":[None]}
    class AutomaticPipeline:
        extractor=type("Extractor",(),{"model_name":"qwen2.5vl:3b"})()
        def process(self,*_args,**_kwargs):
            return {"accepted":True,"qwen_invoked":True,"document_type":"Prescription","confidence":.99,
                "classification":{"document_type":"Prescription"},"extracted_json":qwen,
                "extraction":{"success":True,"attempts":1,"data":qwen,"raw_output":"raw qwen text"},
                "validation":{"valid":True,"requires_review":False},
                "review":{"status":"AUTO_ACCEPT","reasons":[]}}
    db=TestingSession()
    doc=DocumentService(db,AutomaticPipeline()).process_and_store(source,"input.png")
    assert doc.extraction.extraction_json==qwen
    assert doc.extraction_raw.raw_text=="raw qwen text"
    assert doc.verification_runs == []
    assert doc.review_required is False
    db.close(); Base.metadata.drop_all(engine); engine.dispose()
