import sys
from unittest.mock import MagicMock
import pytest

# 1. MOCK MLFLOW BEFORE IMPORTING APP
module_mock = MagicMock()
sys.modules["mlflow"] = module_mock

# Now it is safe to import your app
from app.main import app

@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    return TestClient(app)