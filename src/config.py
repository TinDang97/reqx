"""
Configuration management for enhanced-httpx.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field

# Default configuration file path
DEFAULT_CONFIG_DIR = os.path.expanduser("~/.config/enhanced-httpx")
DEFAULT_CONFIG_FILE = os.path.join(DEFAULT_CONFIG_DIR, "config.json")
DEFAULT_TEMPLATES_DIR = os.path.join(DEFAULT_CONFIG_DIR, "templates")
DEFAULT_HISTORY_FILE = os.path.join(DEFAULT_CONFIG_DIR, "history.json")


class RequestTemplate(BaseModel):
    """Template for saving and reusing requests."""

    name: str
    method: str
    url: str
    headers: Dict[str, str] = Field(default_factory=dict)
    params: Dict[str, Any] = Field(default_factory=dict)
    body: Optional[Union[Dict[str, Any], List[Any]]] = None
    description: Optional[str] = None


class EnhancedHttpxConfig(BaseModel):
    """Configuration for enhanced-httpx client and CLI."""

    default_timeout: float = 30.0
    verify_ssl: bool = True
    follow_redirects: bool = True
    default_headers: Dict[str, str] = Field(default_factory=dict)
    templates_dir: str = DEFAULT_TEMPLATES_DIR
    history_file: Optional[str] = DEFAULT_HISTORY_FILE


def ensure_config_dirs():
    """Ensure configuration directories exist."""
    os.makedirs(DEFAULT_CONFIG_DIR, exist_ok=True)
    os.makedirs(DEFAULT_TEMPLATES_DIR, exist_ok=True)


def load_config(config_file: Optional[str] = None) -> EnhancedHttpxConfig:
    """
    Load configuration from a file.

    Args:
        config_file: Path to the configuration file. If None, the default path is used.

    Returns:
        Loaded configuration
    """
    ensure_config_dirs()

    # Use default if not specified
    if not config_file:
        config_file = DEFAULT_CONFIG_FILE

    # Create default config if it doesn't exist
    if not os.path.exists(config_file):
        config = EnhancedHttpxConfig()
        save_config(config, config_file)
        return config

    # Load existing config
    try:
        with open(config_file, "r") as f:
            config_data = json.load(f)

        return EnhancedHttpxConfig(**config_data)
    except Exception as e:
        # If loading fails, return default config
        print(f"Error loading config: {e}")
        return EnhancedHttpxConfig()


def save_config(config: EnhancedHttpxConfig, config_file: Optional[str] = None) -> bool:
    """
    Save configuration to a file.

    Args:
        config: Configuration to save
        config_file: Path to the configuration file. If None, the default path is used.

    Returns:
        True if successful, False otherwise
    """
    ensure_config_dirs()

    # Use default if not specified
    if not config_file:
        config_file = DEFAULT_CONFIG_FILE

    try:
        # Convert to dict and save as JSON
        config_dict = config.model_dump()
        with open(config_file, "w") as f:
            json.dump(config_dict, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving config: {e}")
        return False


def get_template_path(name: str, config: EnhancedHttpxConfig) -> str:
    """
    Get the path to a template file.

    Args:
        name: Template name
        config: Application configuration

    Returns:
        Template file path
    """
    templates_dir = config.templates_dir
    os.makedirs(templates_dir, exist_ok=True)
    return os.path.join(templates_dir, f"{name}.json")


def save_request_template(template: RequestTemplate, config: EnhancedHttpxConfig) -> bool:
    """
    Save a request template.

    Args:
        template: Template to save
        config: Application configuration

    Returns:
        True if successful, False otherwise
    """
    try:
        template_path = get_template_path(template.name, config)
        template_dict = template.model_dump()

        with open(template_path, "w") as f:
            json.dump(template_dict, f, indent=2)

        return True
    except Exception as e:
        print(f"Error saving template: {e}")
        return False


def load_request_template(name: str, config: EnhancedHttpxConfig) -> Optional[RequestTemplate]:
    """
    Load a request template.

    Args:
        name: Template name
        config: Application configuration

    Returns:
        Loaded template or None if not found
    """
    try:
        template_path = get_template_path(name, config)

        if not os.path.exists(template_path):
            return None

        with open(template_path, "r") as f:
            template_data = json.load(f)

        return RequestTemplate(**template_data)
    except Exception as e:
        print(f"Error loading template: {e}")
        return None


def list_templates(config: EnhancedHttpxConfig) -> List[Tuple[str, Optional[str]]]:
    """
    List all available templates.

    Args:
        config: Application configuration

    Returns:
        List of (template_name, description) tuples
    """
    templates_dir = config.templates_dir
    os.makedirs(templates_dir, exist_ok=True)

    result = []

    for file_name in os.listdir(templates_dir):
        if file_name.endswith(".json"):
            try:
                file_path = os.path.join(templates_dir, file_name)
                with open(file_path, "r") as f:
                    template_data = json.load(f)

                name = template_data.get("name", file_name[:-5])  # Remove .json
                description = template_data.get("description")

                result.append((name, description))
            except Exception:
                # Skip invalid templates
                pass

    return sorted(result)


def delete_template(name: str, config: EnhancedHttpxConfig) -> bool:
    """
    Delete a request template.

    Args:
        name: Template name
        config: Application configuration

    Returns:
        True if successful, False otherwise
    """
    try:
        template_path = get_template_path(name, config)

        if not os.path.exists(template_path):
            return False

        os.remove(template_path)
        return True
    except Exception:
        return False
