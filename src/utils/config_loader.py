"""
Configuration Loader Module.

Loads and merges YAML configuration files with proper precedence:
CLI args > preset yaml > game profile > benchmark_config.yaml defaults
"""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class ConfigLoader:
    """Load and merge YAML configuration files.

    Supports loading from:
        - benchmark_config.yaml: Global defaults
        - presets/*.yaml: Graphics preset overrides
        - games/game_profiles.yaml: Game-specific settings

    Merge precedence (highest wins):
        CLI args > preset yaml > game profile > benchmark_config.yaml
    """

    def __init__(self, config_root: Optional[Path] = None):
        """Initialize the config loader.

        Args:
            config_root: Root directory for config files.
                Defaults to 'config' in the project root.
        """
        if config_root is None:
            # Default to config/ relative to project root
            self.config_root = Path(__file__).parent.parent.parent / "config"
        else:
            self.config_root = Path(config_root)

    def load_yaml(self, filepath: Path) -> Dict[str, Any]:
        """Load a YAML file.

        Args:
            filepath: Path to the YAML file.

        Returns:
            Dict of loaded configuration, or empty dict if file not found.
        """
        if not filepath.exists():
            return {}

        with open(filepath, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return data if data is not None else {}

    def load_benchmark_config(self) -> Dict[str, Any]:
        """Load the main benchmark configuration.

        Returns:
            Dict of global benchmark settings.
        """
        config_path = self.config_root / "benchmark_config.yaml"
        return self.load_yaml(config_path)

    def load_preset(self, preset_name: str) -> Dict[str, Any]:
        """Load a graphics preset configuration.

        Args:
            preset_name: Name of the preset (e.g., 'high', 'ultra').

        Returns:
            Dict of preset settings.

        Raises:
            FileNotFoundError: If preset file doesn't exist.
        """
        preset_path = self.config_root / "presets" / f"{preset_name}.yaml"
        if not preset_path.exists():
            raise FileNotFoundError(f"Preset not found: {preset_name}")
        return self.load_yaml(preset_path)

    def load_game_profile(self, game_name: str) -> Dict[str, Any]:
        """Load a game-specific profile.

        Args:
            game_name: Name of the game.

        Returns:
            Dict of game-specific settings, or empty dict if not found.
        """
        profiles_path = self.config_root / "games" / "game_profiles.yaml"
        profiles = self.load_yaml(profiles_path)

        # Look up game by name (case-insensitive)
        game_key = game_name.lower().replace(" ", "_")
        for key, profile in profiles.get("games", {}).items():
            if key.lower() == game_key:
                return profile

        return {}

    def get_available_presets(self) -> list:
        """Get list of available preset names.

        Returns:
            List of preset names (without .yaml extension).
        """
        presets_dir = self.config_root / "presets"
        if not presets_dir.exists():
            return []

        presets = []
        for filepath in presets_dir.glob("*.yaml"):
            presets.append(filepath.stem)
        return sorted(presets)

    def get_available_games(self) -> list:
        """Get list of available game profiles.

        Returns:
            List of game names.
        """
        profiles_path = self.config_root / "games" / "game_profiles.yaml"
        profiles = self.load_yaml(profiles_path)
        return list(profiles.get("games", {}).keys())

    def merge_configs(self, *configs: Dict[str, Any]) -> Dict[str, Any]:
        """Merge multiple config dicts with later ones taking precedence.

        Args:
            *configs: Config dicts to merge (later overrides earlier).

        Returns:
            Merged configuration dict.
        """
        result: Dict[str, Any] = {}

        for config in configs:
            result = self._deep_merge(result, config)

        return result

    def _deep_merge(
        self,
        base: Dict[str, Any],
        override: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Deep merge two dicts, with override taking precedence.

        Args:
            base: Base dictionary.
            override: Override dictionary (takes precedence).

        Returns:
            Merged dictionary.
        """
        result = base.copy()

        for key, value in override.items():
            if value is None:
                continue

            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def build_benchmark_config(
        self,
        game_name: str,
        preset_name: str,
        cli_overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build final benchmark configuration with proper precedence.

        Merge order (later overrides earlier):
            1. benchmark_config.yaml (global defaults)
            2. game profile (game-specific)
            3. preset (graphics preset)
            4. CLI overrides

        Args:
            game_name: Name of the game.
            preset_name: Name of the graphics preset.
            cli_overrides: Dict of CLI argument overrides.

        Returns:
            Final merged configuration dict.
        """
        # Load each config layer
        global_config = self.load_benchmark_config()
        game_config = self.load_game_profile(game_name)
        preset_config = self.load_preset(preset_name)

        # Merge in order of precedence
        merged = self.merge_configs(
            global_config,
            game_config,
            preset_config,
            cli_overrides or {},
        )

        return merged
