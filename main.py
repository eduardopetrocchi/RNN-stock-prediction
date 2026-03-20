import uvicorn

"""
main.py - Ponto de entrada da aplicação FastAPI.

Uso:
    python main.py
    uvicorn app.app:app --reload --host 0.0.0.0 --port 8000
"""

if __name__ == "__main__":
    uvicorn.run("app.app:app", host="0.0.0.0", port=8000, reload=True)
