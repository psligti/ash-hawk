from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from models import User

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


class UserRepository:
    """Repository for managing User entities in storage.

    This repository provides CRUD operations for User objects with
    proper error handling and logging.

    Example:
        >>> repo = UserRepository()
        >>> user = repo.get_user(1)
        >>> if user:
        ...     print(user.username)
    """

    def __init__(self) -> None:
        """Initialize the repository with an in-memory store."""
        self._store: dict[int, User] = {}
        logger.info("UserRepository initialized")

    def get_user(self, user_id: int) -> User | None:
        """Retrieve a user by ID.

        Args:
            user_id: The unique identifier of the user.

        Returns:
            The User object if found, None otherwise.
        """
        logger.debug(f"Getting user with id={user_id}")
        user = self._store.get(user_id)
        if user is None:
            logger.debug(f"User {user_id} not found")
        return user

    def save_user(self, user: User) -> bool:
        """Save or update a user in the repository.

        Args:
            user: The User object to save.

        Returns:
            True if the save was successful, False otherwise.
        """
        try:
            self._store[user.id] = user
            logger.info(f"Saved user {user.id}")
            return True
        except Exception as e:
            logger.error(f"Failed to save user {user.id}: {e}")
            return False

    def delete_user(self, user_id: int) -> bool:
        """Delete a user from the repository.

        Args:
            user_id: The unique identifier of the user to delete.

        Returns:
            True if the user was deleted, False if the user didn't exist.
        """
        if user_id in self._store:
            del self._store[user_id]
            logger.info(f"Deleted user {user_id}")
            return True
        logger.debug(f"User {user_id} not found for deletion")
        return False

    def list_users(self, limit: int = 100) -> Sequence[User]:
        """List all users up to the specified limit.

        Args:
            limit: Maximum number of users to return. Defaults to 100.

        Returns:
            A sequence of User objects.
        """
        users = list(self._store.values())[:limit]
        logger.debug(f"Listing {len(users)} users (limit={limit})")
        return users
