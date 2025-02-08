# Knowledge Social Media

## Backend

Initial Setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install flask flask-cors networkx faiss-cpu sentence-transformers torch numpy scipy
```

Running the backend:

```bash
cd backend

# From the backend directory
export FLASK_APP=app.py
export FLASK_ENV=development
flask run
```

Testing Backend:

```bash
# from backend/tests
cd backend/tests

python tests/test_backend.py
```

Note: if there are no errors here then its working fine.
