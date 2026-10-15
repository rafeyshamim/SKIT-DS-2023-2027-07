def test_create_and_get_patient(client):
    payload = {
        "medical_record_number": "MRN-TEST-1001",
        "full_name": "Eleanor Vance",
        "date_of_birth": "1975-06-20",
        "gender": "FEMALE",
        "contact_email": "eleanor.vance@example.com",
        "contact_phone": "+1-555-0144",
        "medical_history_notes": "Non-smoker, referred for baseline thoracic evaluation."
    }
    response = client.post("/api/v1/patients", json=payload)
    assert response.status_code == 201
    created = response.json()
    assert created["medical_record_number"] == payload["medical_record_number"]
    assert "id" in created

    # Duplicate check
    dup_response = client.post("/api/v1/patients", json=payload)
    assert dup_response.status_code == 400

    # Get by ID
    get_res = client.get(f"/api/v1/patients/{created['id']}")
    assert get_res.status_code == 200
    assert get_res.json()["full_name"] == "Eleanor Vance"

    # List with search
    list_res = client.get("/api/v1/patients?search=Eleanor")
    assert list_res.status_code == 200
    assert list_res.json()["total"] >= 1
