# MedVision 3D CT Diagnostic Backend

For comprehensive architecture details, API contracts, and deployment guides, refer to:
- [System Architecture](file:///c:/Users/Mohit%20Choudhary/OneDrive/Documents/New%20project/ARCHITECTURE.md)
- [REST API Contract](file:///c:/Users/Mohit%20Choudhary/OneDrive/Documents/New%20project/API_CONTRACT.md)
- [Root README](file:///c:/Users/Mohit%20Choudhary/OneDrive/Documents/New%20project/README.md)

### Running Backend Locally:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Running Test Suite:
```bash
python -m pytest tests -v
```
