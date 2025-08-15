"""User management system for AI Solutions Lab with authentication and access control."""

import hashlib
import secrets
import time
import json
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime, timedelta
import jwt
from enum import Enum

class UserRole(Enum):
    """User roles with different permission levels."""
    ADMIN = "admin"
    POWER_USER = "power_user"
    USER = "user"
    GUEST = "guest"

class Permission(Enum):
    """System permissions."""
    # Search permissions
    SEARCH_BASIC = "search_basic"
    SEARCH_ADVANCED = "search_advanced"
    SEARCH_BATCH = "search_batch"
    
    # RAG permissions
    RAG_BASIC = "rag_basic"
    RAG_ADVANCED = "rag_advanced"
    
    # Document permissions
    DOCUMENT_VIEW = "document_view"
    DOCUMENT_UPLOAD = "document_upload"
    DOCUMENT_DELETE = "document_delete"
    
    # Analytics permissions
    ANALYTICS_VIEW = "analytics_view"
    ANALYTICS_EXPORT = "analytics_export"
    
    # System permissions
    SYSTEM_STATUS = "system_status"
    SYSTEM_CONFIG = "system_config"
    USER_MANAGEMENT = "user_management"

@dataclass
class User:
    """User account information."""
    username: str
    email: str
    password_hash: str
    role: UserRole
    permissions: Set[Permission]
    created_at: float
    last_login: Optional[float] = None
    is_active: bool = True
    profile: Dict[str, Any] = None

@dataclass
class UserSession:
    """User session information."""
    session_id: str
    username: str
    created_at: float
    expires_at: float
    ip_address: str
    user_agent: str
    permissions: Set[Permission]

@dataclass
class LoginAttempt:
    """Login attempt tracking for security."""
    username: str
    timestamp: float
    ip_address: str
    success: bool
    user_agent: str

class UserManager:
    """Comprehensive user management system."""
    
    def __init__(self, data_dir: str = "./data/users"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # User storage
        self.users: Dict[str, User] = {}
        self.active_sessions: Dict[str, UserSession] = {}
        self.login_attempts: List[LoginAttempt] = []
        
        # Security settings
        self.max_login_attempts = 5
        self.lockout_duration = 900  # 15 minutes
        self.session_timeout = 3600  # 1 hour
        self.jwt_secret = self._get_or_generate_secret()
        
        # Role permissions mapping
        self.role_permissions = self._initialize_role_permissions()
        
        # Load existing users
        self._load_users()
        
        # Create default admin user if none exists
        if not self.users:
            self._create_default_admin()
    
    def _get_or_generate_secret(self) -> str:
        """Get existing JWT secret or generate a new one."""
        secret_file = self.data_dir / "jwt_secret.txt"
        
        if secret_file.exists():
            with open(secret_file, 'r') as f:
                return f.read().strip()
        else:
            secret = secrets.token_urlsafe(32)
            with open(secret_file, 'w') as f:
                f.write(secret)
            return secret
    
    def _initialize_role_permissions(self) -> Dict[UserRole, Set[Permission]]:
        """Initialize role-based permissions."""
        return {
            UserRole.ADMIN: set(Permission),  # All permissions
            UserRole.POWER_USER: {
                Permission.SEARCH_BASIC, Permission.SEARCH_ADVANCED, Permission.SEARCH_BATCH,
                Permission.RAG_BASIC, Permission.RAG_ADVANCED,
                Permission.DOCUMENT_VIEW, Permission.DOCUMENT_UPLOAD,
                Permission.ANALYTICS_VIEW, Permission.ANALYTICS_EXPORT,
                Permission.SYSTEM_STATUS
            },
            UserRole.USER: {
                Permission.SEARCH_BASIC, Permission.SEARCH_ADVANCED,
                Permission.RAG_BASIC,
                Permission.DOCUMENT_VIEW,
                Permission.ANALYTICS_VIEW
            },
            UserRole.GUEST: {
                Permission.SEARCH_BASIC,
                Permission.DOCUMENT_VIEW
            }
        }
    
    def _hash_password(self, password: str) -> str:
        """Hash a password using SHA-256."""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify a password against its hash."""
        return self._hash_password(password) == password_hash
    
    def _create_default_admin(self):
        """Create a default admin user."""
        admin_user = User(
            username="admin",
            email="admin@ai-solutions-lab.local",
            password_hash=self._hash_password("admin123"),
            role=UserRole.ADMIN,
            permissions=self.role_permissions[UserRole.ADMIN],
            created_at=time.time(),
            profile={"full_name": "System Administrator"}
        )
        self.users["admin"] = admin_user
        self._save_users()
        print("Default admin user created: username='admin', password='admin123'")
    
    def register_user(self, username: str, email: str, password: str, role: UserRole = UserRole.USER) -> bool:
        """Register a new user."""
        if username in self.users:
            return False
        
        # Validate password strength
        if len(password) < 8:
            return False
        
        # Create user
        user = User(
            username=username,
            email=email,
            password_hash=self._hash_password(password),
            role=role,
            permissions=self.role_permissions[role],
            created_at=time.time(),
            profile={"full_name": username}
        )
        
        self.users[username] = user
        self._save_users()
        return True
    
    def authenticate_user(self, username: str, password: str, ip_address: str, user_agent: str) -> Optional[str]:
        """Authenticate a user and return session ID."""
        # Check if user is locked out
        if self._is_user_locked_out(username, ip_address):
            return None
        
        # Verify credentials
        if username not in self.users or not self._verify_password(password, self.users[username].password_hash):
            self._record_login_attempt(username, ip_address, user_agent, False)
            return None
        
        user = self.users[username]
        if not user.is_active:
            return None
        
        # Record successful login
        self._record_login_attempt(username, ip_address, user_agent, True)
        
        # Update last login
        user.last_login = time.time()
        
        # Create session
        session_id = secrets.token_urlsafe(32)
        session = UserSession(
            session_id=session_id,
            username=username,
            created_at=time.time(),
            expires_at=time.time() + self.session_timeout,
            ip_address=ip_address,
            user_agent=user_agent,
            permissions=user.permissions
        )
        
        self.active_sessions[session_id] = session
        self._save_users()
        
        return session_id
    
    def _is_user_locked_out(self, username: str, ip_address: str) -> bool:
        """Check if a user is locked out due to failed login attempts."""
        recent_attempts = [
            attempt for attempt in self.login_attempts
            if attempt.username == username and attempt.ip_address == ip_address
            and not attempt.success
            and time.time() - attempt.timestamp < self.lockout_duration
        ]
        
        return len(recent_attempts) >= self.max_login_attempts
    
    def _record_login_attempt(self, username: str, ip_address: str, user_agent: str, success: bool):
        """Record a login attempt."""
        attempt = LoginAttempt(
            username=username,
            timestamp=time.time(),
            ip_address=ip_address,
            success=success,
            user_agent=user_agent
        )
        self.login_attempts.append(attempt)
        
        # Keep only last 1000 attempts
        if len(self.login_attempts) > 1000:
            self.login_attempts = self.login_attempts[-1000:]
    
    def validate_session(self, session_id: str) -> Optional[UserSession]:
        """Validate a session and return user session info."""
        if session_id not in self.active_sessions:
            return None
        
        session = self.active_sessions[session_id]
        
        # Check if session has expired
        if time.time() > session.expires_at:
            del self.active_sessions[session_id]
            return None
        
        # Extend session
        session.expires_at = time.time() + self.session_timeout
        
        return session
    
    def logout_user(self, session_id: str) -> bool:
        """Logout a user by invalidating their session."""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            return True
        return False
    
    def get_user_info(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user information (without sensitive data)."""
        if username not in self.users:
            return None
        
        user = self.users[username]
        return {
            'username': user.username,
            'email': user.email,
            'role': user.role.value,
            'permissions': [p.value for p in user.permissions],
            'created_at': user.created_at,
            'last_login': user.last_login,
            'is_active': user.is_active,
            'profile': user.profile
        }
    
    def update_user_profile(self, username: str, profile_data: Dict[str, Any]) -> bool:
        """Update user profile information."""
        if username not in self.users:
            return False
        
        user = self.users[username]
        user.profile.update(profile_data)
        self._save_users()
        return True
    
    def change_user_password(self, username: str, old_password: str, new_password: str) -> bool:
        """Change user password."""
        if username not in self.users:
            return False
        
        user = self.users[username]
        
        # Verify old password
        if not self._verify_password(old_password, user.password_hash):
            return False
        
        # Validate new password
        if len(new_password) < 8:
            return False
        
        # Update password
        user.password_hash = self._hash_password(new_password)
        self._save_users()
        return True
    
    def set_user_role(self, username: str, new_role: UserRole, admin_username: str) -> bool:
        """Change user role (admin only)."""
        if admin_username not in self.users or self.users[admin_username].role != UserRole.ADMIN:
            return False
        
        if username not in self.users:
            return False
        
        user = self.users[username]
        user.role = new_role
        user.permissions = self.role_permissions[new_role]
        self._save_users()
        return True
    
    def deactivate_user(self, username: str, admin_username: str) -> bool:
        """Deactivate a user account (admin only)."""
        if admin_username not in self.users or self.users[admin_username].role != UserRole.ADMIN:
            return False
        
        if username not in self.users:
            return False
        
        user = self.users[username]
        user.is_active = False
        
        # Invalidate all sessions for this user
        sessions_to_remove = [
            session_id for session_id, session in self.active_sessions.items()
            if session.username == username
        ]
        for session_id in sessions_to_remove:
            del self.active_sessions[session_id]
        
        self._save_users()
        return True
    
    def check_permission(self, session_id: str, permission: Permission) -> bool:
        """Check if a user has a specific permission."""
        session = self.validate_session(session_id)
        if not session:
            return False
        
        return permission in session.permissions
    
    def get_active_sessions(self) -> List[Dict[str, Any]]:
        """Get information about all active sessions."""
        sessions = []
        current_time = time.time()
        
        for session_id, session in self.active_sessions.items():
            if current_time <= session.expires_at:
                sessions.append({
                    'session_id': session_id,
                    'username': session.username,
                    'created_at': session.created_at,
                    'expires_at': session.expires_at,
                    'ip_address': session.ip_address,
                    'user_agent': session.user_agent,
                    'permissions': [p.value for p in session.permissions]
                })
        
        return sessions
    
    def cleanup_expired_sessions(self):
        """Remove expired sessions."""
        current_time = time.time()
        expired_sessions = [
            session_id for session_id, session in self.active_sessions.items()
            if current_time > session.expires_at
        ]
        
        for session_id in expired_sessions:
            del self.active_sessions[session_id]
        
        if expired_sessions:
            print(f"Cleaned up {len(expired_sessions)} expired sessions")
    
    def generate_jwt_token(self, session_id: str) -> Optional[str]:
        """Generate a JWT token for a session."""
        session = self.validate_session(session_id)
        if not session:
            return None
        
        payload = {
            'session_id': session_id,
            'username': session.username,
            'permissions': [p.value for p in session.permissions],
            'exp': session.expires_at,
            'iat': time.time()
        }
        
        return jwt.encode(payload, self.jwt_secret, algorithm='HS256')
    
    def validate_jwt_token(self, token: str) -> Optional[UserSession]:
        """Validate a JWT token and return user session."""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
            session_id = payload.get('session_id')
            
            if session_id:
                return self.validate_session(session_id)
            
        except jwt.ExpiredSignatureError:
            print("JWT token expired")
        except jwt.InvalidTokenError:
            print("Invalid JWT token")
        
        return None
    
    def _load_users(self):
        """Load users from disk."""
        try:
            users_file = self.data_dir / "users.json"
            if users_file.exists():
                with open(users_file, 'r') as f:
                    data = json.load(f)
                
                for username, user_data in data.items():
                    # Convert role string back to enum
                    user_data['role'] = UserRole(user_data['role'])
                    
                    # Convert permissions back to enum set
                    user_data['permissions'] = {Permission(p) for p in user_data['permissions']}
                    
                    user = User(**user_data)
                    self.users[username] = user
                
                print(f"Loaded {len(self.users)} users from disk")
                
        except Exception as e:
            print(f"Warning: Could not load users: {e}")
    
    def _save_users(self):
        """Save users to disk."""
        try:
            users_file = self.data_dir / "users.json"
            
            # Convert users to serializable format
            users_data = {}
            for username, user in self.users.items():
                user_dict = asdict(user)
                user_dict['role'] = user.role.value
                user_dict['permissions'] = [p.value for p in user.permissions]
                users_data[username] = user_dict
            
            with open(users_file, 'w') as f:
                json.dump(users_data, f, indent=2, default=str)
                
        except Exception as e:
            print(f"Warning: Could not save users: {e}")


def main():
    """Demo the user management system."""
    print("User Management System Demo")
    print("=" * 40)
    
    # Create user manager
    user_manager = UserManager()
    
    # Register some test users
    print("Registering test users...")
    user_manager.register_user("alice", "alice@example.com", "password123", UserRole.POWER_USER)
    user_manager.register_user("bob", "bob@example.com", "password456", UserRole.USER)
    user_manager.register_user("guest", "guest@example.com", "guest123", UserRole.GUEST)
    
    # Test authentication
    print("\nTesting authentication...")
    session_id = user_manager.authenticate_user("alice", "password123", "192.168.1.100", "Mozilla/5.0")
    
    if session_id:
        print(f"Alice authenticated successfully. Session ID: {session_id[:8]}...")
        
        # Check permissions
        can_search_advanced = user_manager.check_permission(session_id, Permission.SEARCH_ADVANCED)
        can_manage_users = user_manager.check_permission(session_id, Permission.USER_MANAGEMENT)
        
        print(f"Can perform advanced search: {can_search_advanced}")
        print(f"Can manage users: {can_manage_users}")
        
        # Generate JWT token
        jwt_token = user_manager.generate_jwt_token(session_id)
        print(f"JWT token generated: {jwt_token[:20]}...")
        
        # Validate JWT token
        session = user_manager.validate_jwt_token(jwt_token)
        if session:
            print(f"JWT token validated for user: {session.username}")
        
        # Logout
        user_manager.logout_user(session_id)
        print("Alice logged out")
    
    # Show user info
    print("\nUser information:")
    for username in ["admin", "alice", "bob", "guest"]:
        user_info = user_manager.get_user_info(username)
        if user_info:
            print(f"{username}: {user_info['role']} role, {len(user_info['permissions'])} permissions")
    
    # Show active sessions
    print(f"\nActive sessions: {len(user_manager.get_active_sessions())}")
    
    # Cleanup
    user_manager.cleanup_expired_sessions()


if __name__ == "__main__":
    main()
