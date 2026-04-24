"""
Run the FastAPI server.

Usage:
    python run_api.py
    # Then open in your browser:
    #   http://localhost:8000         -- the web UI
    #   http://localhost:8000/docs    -- auto-generated Swagger docs
    #   http://localhost:8000/health  -- health check
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import uvicorn


HOST = "127.0.0.1"  # Browser-friendly. Use 0.0.0.0 in Docker / production.
PORT = 8000


def main() -> None:
    # Print a banner with the real URL so there's no ambiguity.
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  Acquirer Engine -- open in your browser:")
    print(f"")
    print(f"    Web UI:       http://localhost:{PORT}")
    print(f"    API docs:     http://localhost:{PORT}/docs")
    print(f"    Health check: http://localhost:{PORT}/health")
    print(f"{bar}\n")

    uvicorn.run(
        "acquirer_engine.api:app",
        host=HOST,
        port=PORT,
        reload=False,  # Set True for dev
        log_level="info",
    )


if __name__ == "__main__":
    main()
