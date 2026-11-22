import os
import warnings
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Suppress Starlette/httpx test deprecation warnings
warnings.filterwarnings("ignore", message=".*StarletteDeprecationWarning.*")
warnings.filterwarnings("ignore", message=".*Using `httpx` with `starlette.testclient` is deprecated.*")

# Set test environment
os.environ["ENVIRONMENT"] = "test"
os.environ["DEBUG"] = "false"
os.environ["DATABASE_URL"] = "sqlite:///./test_medvision.db"

from fastapi.testclient import TestClient
from app.main import app
from app.db.session import Base, get_db
from app.core.config import settings

# Test database engine
engine = create_engine(
    "sqlite:///./test_medvision.db",
    connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db


@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    engine.dispose()
    for db_path in ["./test_medvision.db", "./backend/test_medvision.db"]:
        if os.path.exists(db_path):
            try:
                os.remove(db_path)
            except Exception:
                pass
    Base.metadata.create_all(bind=engine)
    yield
    try:
        Base.metadata.drop_all(bind=engine)
    except Exception:
        pass
    engine.dispose()
    for db_path in ["./test_medvision.db", "./backend/test_medvision.db"]:
        if os.path.exists(db_path):
            try:
                os.remove(db_path)
            except Exception:
                pass


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client
