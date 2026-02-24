import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models import User


class TestUserRepository:
    def test_get_user_exists(self, repo):
        user = repo.get_user(1)
        assert user is not None
        assert user.id == 1
        assert isinstance(user, User)

    def test_get_user_not_exists(self, repo):
        user = repo.get_user(99999)
        assert user is None

    def test_save_user_new(self, repo):
        user = User(
            id=100,
            username="newuser",
            email="new@example.com",
            created_at=datetime.now(),
        )
        result = repo.save_user(user)
        assert result is True

        saved = repo.get_user(100)
        assert saved is not None
        assert saved.username == "newuser"

    def test_save_user_update(self, repo):
        user = User(
            id=1,
            username="updated",
            email="updated@example.com",
            created_at=datetime.now(),
        )
        result = repo.save_user(user)
        assert result is True

        saved = repo.get_user(1)
        assert saved.username == "updated"

    def test_delete_user_exists(self, repo):
        result = repo.delete_user(1)
        assert result is True

        deleted = repo.get_user(1)
        assert deleted is None

    def test_delete_user_not_exists(self, repo):
        result = repo.delete_user(99999)
        assert result is False

    def test_list_users_default_limit(self, repo):
        users = repo.list_users()
        assert isinstance(users, list)
        assert len(users) <= 100

    def test_list_users_custom_limit(self, repo):
        users = repo.list_users(limit=10)
        assert isinstance(users, list)
        assert len(users) <= 10

    def test_list_users_empty(self, empty_repo):
        users = empty_repo.list_users()
        assert users == []


@pytest.fixture
def repo():
    from user_repository import UserRepository

    return UserRepository()


@pytest.fixture
def empty_repo():
    from user_repository import UserRepository

    repo = UserRepository()
    return repo
