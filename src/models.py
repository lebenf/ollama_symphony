import re
from dataclasses import dataclass


@dataclass
class User:
    id: int
    name: str
    email: str


_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def validate_email(email: str) -> bool:
    return bool(_EMAIL_RE.match(email))
