"""
Vercel Serverless Function Entrypoint
Exposes the FastAPI app instance for Vercel serverless deployment.
"""
from app.main import app
