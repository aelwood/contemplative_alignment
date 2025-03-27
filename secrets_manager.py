import os
import json
from typing import Optional

DEFAULT_SECRETS_PATH = "secrets.json"

def load_api_key(key_name: str = "openai_api_key", secrets_path: Optional[str] = None) -> Optional[str]:
    """
    Load an API key from a secrets file or environment variable.
    
    Args:
        key_name: Name of the API key in the secrets file
        secrets_path: Path to the secrets file. If None, uses DEFAULT_SECRETS_PATH
        
    Returns:
        The API key if found, None otherwise
    """
    # Try to load from secrets file first
    secrets_path = secrets_path or DEFAULT_SECRETS_PATH
    
    if os.path.exists(secrets_path):
        try:
            with open(secrets_path, 'r') as f:
                secrets = json.load(f)
                if key_name in secrets:
                    return secrets[key_name]
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Failed to load secrets from {secrets_path}: {e}")
    
    # Fall back to environment variable (convert snake_case to UPPER_SNAKE_CASE)
    env_var_name = key_name.upper()
    return os.environ.get(env_var_name)

def save_api_key(api_key: str, key_name: str = "openai_api_key", secrets_path: Optional[str] = None) -> bool:
    """
    Save an API key to a secrets file.
    
    Args:
        api_key: The API key to save
        key_name: Name of the API key in the secrets file
        secrets_path: Path to the secrets file. If None, uses DEFAULT_SECRETS_PATH
        
    Returns:
        True if successful, False otherwise
    """
    secrets_path = secrets_path or DEFAULT_SECRETS_PATH
    
    # Load existing secrets if file exists
    secrets = {}
    if os.path.exists(secrets_path):
        try:
            with open(secrets_path, 'r') as f:
                secrets = json.load(f)
        except (json.JSONDecodeError, IOError):
            # If we can't load the existing file, start with an empty dict
            secrets = {}
    
    # Update the API key
    secrets[key_name] = api_key
    
    # Save the updated secrets
    try:
        with open(secrets_path, 'w') as f:
            json.dump(secrets, f, indent=2)
        
        # Set file permissions to restrict access (Unix-like systems only)
        try:
            os.chmod(secrets_path, 0o600)  # Read/write for owner only
        except:
            # On Windows or if chmod fails, just print a warning
            print(f"Warning: Could not set restrictive permissions on {secrets_path}")
        
        return True
    except IOError as e:
        print(f"Error: Failed to save API key to {secrets_path}: {e}")
        return False 