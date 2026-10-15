"""
MedVision 3D - Unified Full-Stack System Launcher
Starts both the FastAPI Backend (port 8000) and the React Frontend (port 3000) concurrently.
"""
import os
import sys
import subprocess
import time
import signal

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(ROOT_DIR, "backend")
FRONTEND_DIR = os.path.join(ROOT_DIR, "frontend")

def run():
    print("=" * 70)
    print("   MEDVISION 3D - LAUNCHING END-TO-END SYSTEM (REACT + FASTAPI)")
    print("=" * 70)

    # 1. Start Backend
    python_exe = os.path.join(ROOT_DIR, "venv", "Scripts", "python.exe")
    if not os.path.exists(python_exe):
        python_exe = sys.executable

    print("\n[1/2] Starting FastAPI Backend on http://127.0.0.1:8000 ...")
    backend_cmd = [
        python_exe, "-m", "uvicorn", "app.main:app",
        "--app-dir", "backend",
        "--host", "127.0.0.1",
        "--port", "8000",
        "--reload"
    ]
    backend_proc = subprocess.Popen(backend_cmd, cwd=ROOT_DIR)

    # Allow backend 2 seconds to initialize
    time.sleep(2)

    # 2. Start Frontend
    print("[2/2] Starting React Vite Frontend on http://localhost:3000 ...")
    frontend_cmd = ["npm.cmd" if os.name == "nt" else "npm", "run", "dev"]
    frontend_proc = subprocess.Popen(frontend_cmd, cwd=FRONTEND_DIR)

    print("\n>>> System is LIVE! <<<")
    print("  -> React UI:        http://localhost:3000")
    print("  -> FastAPI Docs:    http://127.0.0.1:8000/docs")
    print("  -> Health Check:    http://127.0.0.1:8000/api/v1/health")
    print("\nPress Ctrl+C to terminate both servers.")

    try:
        while True:
            time.sleep(1)
            if backend_proc.poll() is not None:
                print("Backend terminated unexpectedly.")
                break
            if frontend_proc.poll() is not None:
                print("Frontend terminated unexpectedly.")
                break
    except KeyboardInterrupt:
        print("\nShutting down services...")
    finally:
        for p in [backend_proc, frontend_proc]:
            try:
                p.terminate()
                p.wait(timeout=2)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
        print("All servers stopped cleanly.")

if __name__ == "__main__":
    run()
