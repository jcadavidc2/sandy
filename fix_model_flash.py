#!/usr/bin/env python3
"""Update OpenCLAW config to use Gemini 2.5 Flash (not Lite).

Gemini 2.5 Flash (openrouter/google/gemini-2.5-flash-preview-05-20) is the
latest stable Flash model. It's cheap ($0.15/M input, $0.60/M output) but
significantly better at tool calling than Lite.

This script updates the MCP client configuration to use the correct model.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# The model we want to use
TARGET_MODEL = "openrouter/google/gemini-2.5-flash-preview-05-20"

# Common config file locations for OpenCLAW / Claude Desktop / Cline
CONFIG_PATHS = [
    Path.home() / ".config" / "openclaw" / "config.json",
    Path.home() / ".openclaw" / "config.json",
    Path.home() / ".config" / "claude-desktop" / "claude_desktop_config.json",
    Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json",
    Path(".") / ".openclaw.json",
    Path(".") / "openclaw.json",
]


def find_config() -> Path | None:
    """Find the first existing config file."""
    for path in CONFIG_PATHS:
        if path.exists():
            return path
    return None


def update_config(config_path: Path) -> bool:
    """Update the model in the config file to Gemini 2.5 Flash."""
    try:
        with open(config_path) as f:
            config = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"Error reading {config_path}: {e}")
        return False

    # Update model references
    updated = False

    # Check for model field at top level
    if "model" in config:
        old_model = config["model"]
        config["model"] = TARGET_MODEL
        if old_model != TARGET_MODEL:
            print(f"  Updated top-level model: {old_model} -> {TARGET_MODEL}")
            updated = True

    # Check for model in mcpServers or similar nested configs
    if "mcpServers" in config:
        for server_name, server_config in config["mcpServers"].items():
            if isinstance(server_config, dict) and "model" in server_config:
                old_model = server_config["model"]
                server_config["model"] = TARGET_MODEL
                if old_model != TARGET_MODEL:
                    print(f"  Updated {server_name} model: {old_model} -> {TARGET_MODEL}")
                    updated = True

    # Check for model in provider/llm config sections
    for key in ("provider", "llm", "ai"):
        if key in config and isinstance(config[key], dict):
            if "model" in config[key]:
                old_model = config[key]["model"]
                config[key]["model"] = TARGET_MODEL
                if old_model != TARGET_MODEL:
                    print(f"  Updated {key}.model: {old_model} -> {TARGET_MODEL}")
                    updated = True

    if not updated:
        # If no model field found, add it at top level
        config["model"] = TARGET_MODEL
        print(f"  Added model field: {TARGET_MODEL}")
        updated = True

    # Write back
    try:
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
            f.write("\n")
        return True
    except OSError as e:
        print(f"Error writing {config_path}: {e}")
        return False


def main() -> int:
    print(f"Upgrading model to: {TARGET_MODEL}")
    print(f"  (Gemini 2.5 Flash - $0.15/M input, excellent tool calling)")
    print()

    # Allow explicit path via argument or env var
    explicit_path = None
    if len(sys.argv) > 1:
        explicit_path = Path(sys.argv[1])
    elif os.environ.get("OPENCLAW_CONFIG"):
        explicit_path = Path(os.environ["OPENCLAW_CONFIG"])

    if explicit_path:
        if not explicit_path.exists():
            print(f"Config file not found: {explicit_path}")
            return 1
        config_path = explicit_path
    else:
        config_path = find_config()

    if config_path is None:
        print("No config file found. Searched:")
        for p in CONFIG_PATHS:
            print(f"  - {p}")
        print()
        print("Create one or pass the path as an argument:")
        print(f"  python fix_model_flash.py /path/to/config.json")
        print()
        print("Or set OPENCLAW_CONFIG environment variable.")

        # Create a minimal config as a starting point
        default_path = Path(".") / ".openclaw.json"
        minimal_config = {
            "model": TARGET_MODEL,
            "mcpServers": {
                "sandy": {
                    "command": "python",
                    "args": ["-m", "sandy.mcp"],
                }
            },
        }
        with open(default_path, "w") as f:
            json.dump(minimal_config, f, indent=2)
            f.write("\n")
        print(f"\nCreated minimal config at: {default_path}")
        return 0

    print(f"Found config: {config_path}")
    if update_config(config_path):
        print(f"\n✅ Model updated to {TARGET_MODEL}")
        return 0
    else:
        print("\n❌ Failed to update config")
        return 1


if __name__ == "__main__":
    sys.exit(main())
