Decision Support Service (DSS)

This module exposes a FastAPI endpoint to get operational recommendations for a railway scenario.

File: `dss.py`

- Endpoint: `POST /recommend`
- Payload: JSON with a top-level `data` object containing the scenario dictionary.
- Models: looks for `./models/conflict_model.pkl` and `./models/delay_model.pkl` relative to the backend folder.

Run locally:

```powershell
pip install -r requirements.txt
uvicorn dss:app --reload --port 8001
```

If models are missing, the service will return placeholder predictions (0.0) to allow integration testing.
