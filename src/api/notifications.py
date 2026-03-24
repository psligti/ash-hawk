"""
Notification Service Module

Provides functionality for managing user notifications.
"""

from datetime import UTC, datetime, timezone
from typing import Any, Dict, List, Optional


class NotificationService:
    """
    Service class for handling notification operations.
    """
    
    def __init__(self, storage_backend: Optional[Any] = None):
        """
        Initialize the NotificationService.
        
        Args:
            storage_backend: Optional storage backend for persisting notifications
        """
        self.storage_backend = storage_backend
        self._notifications: List[Dict[str, Any]] = []
    
    def create_notification(
        self,
        user_id: str,
        message: str,
        notification_type: str = "info",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a new notification.
        
        Args:
            user_id: The ID of the user to notify
            message: The notification message
            notification_type: Type of notification (info, warning, error, success)
            metadata: Optional additional metadata
            
        Returns:
            Dictionary containing the created notification details
        """
        notification = {
            "id": self._generate_id(),
            "user_id": user_id,
            "message": message,
            "type": notification_type,
            "metadata": metadata or {},
            "created_at": datetime.now(UTC).isoformat(),
            "read": False
        }
        
        self._notifications.append(notification)
        
        if self.storage_backend:
            self.storage_backend.save(notification)
        
        return notification
    
    def get_user_notifications(
        self,
        user_id: str,
        include_read: bool = True,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve notifications for a specific user.
        
        Args:
            user_id: The ID of the user
            include_read: Whether to include read notifications
            limit: Maximum number of notifications to return
            
        Returns:
            List of notification dictionaries
        """
        notifications = [
            n for n in self._notifications
            if n["user_id"] == user_id and (include_read or not n["read"])
        ]
        
        # Sort by created_at descending (newest first)
        notifications.sort(key=lambda x: x["created_at"], reverse=True)
        
        if limit:
            notifications = notifications[:limit]
        
        return notifications
    
    def mark_as_read(self, notification_id: str) -> bool:
        """
        Mark a notification as read.
        
        Args:
            notification_id: The ID of the notification
            
        Returns:
            True if successful, False if notification not found
        """
        for notification in self._notifications:
            if notification["id"] == notification_id:
                notification["read"] = True
                if self.storage_backend:
                    self.storage_backend.update(notification_id, {"read": True})
                return True
        return False
    
    def delete_notification(self, notification_id: str) -> bool:
        """
        Delete a notification.
        
        Args:
            notification_id: The ID of the notification
            
        Returns:
            True if successful, False if notification not found
        """
        for i, notification in enumerate(self._notifications):
            if notification["id"] == notification_id:
                del self._notifications[i]
                if self.storage_backend:
                    self.storage_backend.delete(notification_id)
                return True
        return False
    
    def _generate_id(self) -> str:
        """
        Generate a unique notification ID.
        
        Returns:
            A unique ID string
        """
        timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
        return f"notif_{timestamp}_{len(self._notifications)}"
