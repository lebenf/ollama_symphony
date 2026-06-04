import pytest
from src.models import User, validate_email


def test_user_dataclass_fields():
    user = User(id=1, name="Alice", email="alice@example.com")
    assert user.id == 1
    assert user.name == "Alice"
    assert user.email == "alice@example.com"


def test_user_equality():
    u1 = User(id=2, name="Bob", email="bob@example.com")
    u2 = User(id=2, name="Bob", email="bob@example.com")
    assert u1 == u2


@pytest.mark.parametrize("email", [
    "user@example.com",
    "user.name+tag@sub.domain.org",
    "a@b.co",
])
def test_validate_email_valid(email):
    assert validate_email(email) is True


@pytest.mark.parametrize("email", [
    "notanemail",
    "@nodomain.com",
    "missing@tld",
    "spaces in@email.com",
    "",
])
def test_validate_email_invalid(email):
    assert validate_email(email) is False
