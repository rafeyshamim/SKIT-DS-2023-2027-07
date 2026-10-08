def test_health_check(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ONLINE"
    assert "relational_db" in data["database"]
    assert data["ml_inference"]["is_loaded"] is True
