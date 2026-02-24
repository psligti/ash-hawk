from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class User:
    id: int
    username: str
    email: str
    created_at: datetime
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "User":
        return cls(
            id=data["id"],
            username=data["username"],
            email=data["email"],
            created_at=datetime.fromisoformat(data["created_at"]),
            metadata=data.get("metadata"),
        )
