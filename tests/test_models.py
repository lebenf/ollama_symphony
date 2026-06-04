from src.models import User


def test_user_dataclass_fields():
    user = User(id=1, name="Alice", email="alice@example.com")
    assert user.id == 1
    assert user.name == "Alice"
    assert user.email == "alice@example.com"


def test_user_equality():
    u1 = User(id=2, name="Bob", email="bob@example.com")
    u2 = User(id=2, name="Bob", email="bob@example.com")
    assert u1 == u2
