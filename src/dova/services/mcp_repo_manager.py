"""
MCP Repository Manager.

Manages local clones of MCP server repositories with automatic updates.
"""

import asyncio
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class MCPRepoConfig:
    """Configuration for a cloned MCP server repository."""

    name: str
    repo_url: str
    local_path: Path
    command: str  # Command to run the server
    args: list[str] = field(default_factory=list)
    env_vars: dict[str, str] = field(default_factory=dict)
    auto_update: bool = True
    last_updated: datetime | None = None


# Default MCP repos to manage
DEFAULT_MCP_REPOS: dict[str, MCPRepoConfig] = {
    "arxiv": MCPRepoConfig(
        name="arxiv",
        repo_url="https://github.com/blazickjp/arxiv-mcp-server.git",
        local_path=Path.home() / ".dova" / "mcp-servers" / "arxiv-mcp-server",
        command="uv",
        args=["run", "arxiv-mcp-server", "--storage-path", str(Path.home() / ".dova" / "arxiv-papers")],
    ),
}


class MCPRepoManager:
    """
    Manages MCP server repository clones.

    Handles cloning, updating, and running MCP servers from git repos.
    """

    def __init__(self, base_path: Path | None = None):
        self.base_path = base_path or Path.home() / ".dova" / "mcp-servers"
        self._repos: dict[str, MCPRepoConfig] = dict(DEFAULT_MCP_REPOS)
        self._logger = logger.bind(service="mcp_repo_manager")

    def get_repo(self, name: str) -> MCPRepoConfig | None:
        """Get a repo configuration by name."""
        return self._repos.get(name)

    def list_repos(self) -> list[MCPRepoConfig]:
        """List all managed repos."""
        return list(self._repos.values())

    def register_repo(self, config: MCPRepoConfig) -> None:
        """Register a new MCP repo to manage."""
        self._repos[config.name] = config
        self._logger.info("repo_registered", name=config.name, url=config.repo_url)

    async def setup_repo(self, name: str) -> bool:
        """
        Clone or update a repository.

        Args:
            name: Repository name

        Returns:
            True if setup succeeded
        """
        config = self._repos.get(name)
        if not config:
            self._logger.error("repo_not_found", name=name)
            return False

        try:
            # Ensure base path exists
            config.local_path.parent.mkdir(parents=True, exist_ok=True)

            if config.local_path.exists():
                # Update existing repo
                return await self._update_repo(config)
            else:
                # Clone new repo
                return await self._clone_repo(config)

        except Exception as e:
            self._logger.error("setup_failed", name=name, error=str(e))
            return False

    async def _clone_repo(self, config: MCPRepoConfig) -> bool:
        """Clone a repository."""
        self._logger.info("cloning_repo", name=config.name, url=config.repo_url)

        process = await asyncio.create_subprocess_exec(
            "git", "clone", config.repo_url, str(config.local_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            self._logger.error(
                "clone_failed",
                name=config.name,
                stderr=stderr.decode() if stderr else "",
            )
            return False

        # Install dependencies if pyproject.toml exists
        if (config.local_path / "pyproject.toml").exists():
            await self._install_deps(config)

        config.last_updated = datetime.utcnow()
        self._logger.info("clone_succeeded", name=config.name)
        return True

    async def _update_repo(self, config: MCPRepoConfig) -> bool:
        """Update an existing repository."""
        self._logger.info("updating_repo", name=config.name)

        # Fetch and pull
        process = await asyncio.create_subprocess_exec(
            "git", "-C", str(config.local_path), "pull", "--ff-only",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            self._logger.error(
                "update_failed",
                name=config.name,
                stderr=stderr.decode() if stderr else "",
            )
            return False

        # Check if there were actual changes
        output = stdout.decode() if stdout else ""
        if "Already up to date" not in output:
            # Changes detected, reinstall deps
            if (config.local_path / "pyproject.toml").exists():
                await self._install_deps(config)

        config.last_updated = datetime.utcnow()
        self._logger.info("update_succeeded", name=config.name)
        return True

    async def _install_deps(self, config: MCPRepoConfig) -> bool:
        """Install dependencies for a repo."""
        self._logger.info("installing_deps", name=config.name)

        process = await asyncio.create_subprocess_exec(
            "uv", "pip", "install", "-e", str(config.local_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            self._logger.warning(
                "deps_install_failed",
                name=config.name,
                stderr=stderr.decode() if stderr else "",
            )
            return False

        return True

    async def setup_all(self) -> dict[str, bool]:
        """Setup all registered repos."""
        results = {}
        for name in self._repos:
            results[name] = await self.setup_repo(name)
        return results

    async def update_all(self) -> dict[str, bool]:
        """Update all repos that have auto_update enabled."""
        results = {}
        for name, config in self._repos.items():
            if config.auto_update:
                results[name] = await self.setup_repo(name)
        return results

    def get_server_command(self, name: str) -> tuple[str, list[str], dict[str, str]] | None:
        """
        Get the command to run an MCP server.

        Returns:
            Tuple of (command, args, env_vars) or None if not found
        """
        config = self._repos.get(name)
        if not config or not config.local_path.exists():
            return None

        # Build args with directory path
        args = ["--directory", str(config.local_path)] + config.args

        return (config.command, args, config.env_vars)

    def is_repo_installed(self, name: str) -> bool:
        """Check if a repo is installed."""
        config = self._repos.get(name)
        if not config:
            return False
        return config.local_path.exists()


# Global instance
_manager: MCPRepoManager | None = None


def get_mcp_repo_manager() -> MCPRepoManager:
    """Get the global MCP repo manager instance."""
    global _manager
    if _manager is None:
        _manager = MCPRepoManager()
    return _manager


async def setup_mcp_repos() -> dict[str, bool]:
    """Setup all MCP repos. Called during startup."""
    manager = get_mcp_repo_manager()
    return await manager.setup_all()


async def update_mcp_repos() -> dict[str, bool]:
    """Update all MCP repos. Called by heartbeat task."""
    manager = get_mcp_repo_manager()
    return await manager.update_all()
