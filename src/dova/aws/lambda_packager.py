"""Lambda Packager for DOVA deployment.

Packages DOVA source code and dependencies into a ZIP file for Lambda deployment.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class PackageResult:
    """Result of Lambda package creation."""

    success: bool
    zip_path: str | None = None
    size_bytes: int = 0
    error: str | None = None


class LambdaPackager:
    """Packages DOVA for Lambda deployment."""

    # Maximum unzipped size for Lambda (250 MB)
    MAX_UNZIPPED_SIZE = 250 * 1024 * 1024

    # Dependencies to exclude (too large or not needed in Lambda)
    EXCLUDE_PACKAGES = {
        "pip",
        "setuptools",
        "wheel",
        "pytest",
        "mypy",
        "ruff",
        "pre-commit",
        "moto",
        "mkdocs",
        "mkdocstrings",
    }

    # Patterns to exclude from the package
    EXCLUDE_PATTERNS = {
        "__pycache__",
        "*.pyc",
        "*.pyo",
        "*.egg-info",
        ".git",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "tests",
        "test_*",
        "*_test.py",
        "docs",
        "*.md",
        "*.rst",
    }

    def __init__(self) -> None:
        self._logger = logger.bind(component="lambda_packager")
        self._dova_root = self._find_dova_root()

    def _find_dova_root(self) -> Path:
        """Find the DOVA package root directory."""
        # Try to find via import
        try:
            import dova

            dova_init = Path(dova.__file__)
            return dova_init.parent
        except ImportError:
            pass

        # Fallback: look relative to this file
        return Path(__file__).parent.parent

    def create_package(self, output_dir: str | None = None) -> PackageResult:
        """Create a Lambda deployment package.

        Args:
            output_dir: Directory to store the ZIP file (default: temp dir)

        Returns:
            PackageResult with the path to the ZIP file
        """
        self._logger.info("creating_lambda_package", dova_root=str(self._dova_root))

        try:
            # Create temp directory for building
            with tempfile.TemporaryDirectory() as build_dir:
                build_path = Path(build_dir)

                # Step 1: Copy DOVA source
                self._copy_dova_source(build_path)

                # Step 2: Install dependencies
                self._install_dependencies(build_path)

                # Step 3: Create Lambda handler bootstrap
                self._create_bootstrap(build_path)

                # Step 4: Calculate size
                total_size = self._calculate_size(build_path)
                self._logger.info("package_size", size_mb=total_size / (1024 * 1024))

                if total_size > self.MAX_UNZIPPED_SIZE:
                    return PackageResult(
                        success=False,
                        error=f"Package size ({total_size / (1024*1024):.1f} MB) exceeds "
                        f"Lambda limit ({self.MAX_UNZIPPED_SIZE / (1024*1024):.1f} MB)",
                    )

                # Step 5: Create ZIP file
                out_path = Path(output_dir) if output_dir else Path(tempfile.gettempdir())

                out_path.mkdir(parents=True, exist_ok=True)
                zip_path = out_path / "dova-lambda.zip"

                self._create_zip(build_path, zip_path)

                zip_size = zip_path.stat().st_size
                self._logger.info(
                    "package_created",
                    path=str(zip_path),
                    zip_size_mb=zip_size / (1024 * 1024),
                    unzipped_size_mb=total_size / (1024 * 1024),
                )

                return PackageResult(
                    success=True,
                    zip_path=str(zip_path),
                    size_bytes=zip_size,
                )

        except Exception as e:
            self._logger.exception("package_creation_failed", error=str(e))
            return PackageResult(success=False, error=str(e))

    def _copy_dova_source(self, build_path: Path) -> None:
        """Copy DOVA source code to build directory."""
        self._logger.info("copying_dova_source")

        dest = build_path / "dova"
        dest.mkdir(parents=True, exist_ok=True)

        def ignore_patterns(directory: str, files: list[str]) -> set[str]:  # noqa: ARG001
            ignored = set()
            for f in files:
                # Ignore patterns
                for pattern in self.EXCLUDE_PATTERNS:
                    if "*" in pattern:
                        # Glob pattern
                        import fnmatch

                        if fnmatch.fnmatch(f, pattern):
                            ignored.add(f)
                    else:
                        # Exact match
                        if f == pattern:
                            ignored.add(f)
                # Ignore .pyc files
                if f.endswith((".pyc", ".pyo")):
                    ignored.add(f)
                # Ignore __pycache__ directories
                if f == "__pycache__":
                    ignored.add(f)
            return ignored

        shutil.copytree(
            self._dova_root,
            dest,
            ignore=ignore_patterns,
            dirs_exist_ok=True,
        )

    def _install_dependencies(self, build_path: Path) -> None:
        """Install Python dependencies to build directory.

        Uses a two-pass approach:
        1. Install all packages normally
        2. Reinstall binary packages with Lambda-compatible manylinux wheels
           to ensure C extensions are correct for Lambda's Amazon Linux 2
        """
        self._logger.info("installing_dependencies")

        # Get requirements
        requirements = self._get_runtime_requirements()

        if not requirements:
            self._logger.warning("no_requirements_found")
            return

        # First pass: Install all requirements
        self._logger.info("installing_all_packages")
        req_file = build_path / "requirements.txt"
        req_file.write_text("\n".join(requirements))

        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-r",
                str(req_file),
                "-t",
                str(build_path),
                "--no-compile",
                "--quiet",
            ],
            check=True,
            capture_output=True,
        )
        req_file.unlink()

        # Packages with C extensions that MUST use manylinux wheels for Lambda
        # Map package names to their directory names
        binary_packages = {
            "pydantic-core": "pydantic_core",
            "cryptography": "cryptography",
            "aiohttp": "aiohttp",
            "multidict": "multidict",
            "yarl": "yarl",
            "frozenlist": "frozenlist",
            "charset-normalizer": "charset_normalizer",
            "cffi": "cffi",
        }

        # Remove existing binary packages before reinstalling
        # This ensures the manylinux wheels properly replace the local binaries
        self._logger.info("removing_local_binary_packages")
        for pkg_name, dir_name in binary_packages.items():
            pkg_dir = build_path / dir_name
            if pkg_dir.exists():
                shutil.rmtree(pkg_dir)
            # Also remove .libs directories that some packages create
            libs_dir = build_path / f"{dir_name}.libs"
            if libs_dir.exists():
                shutil.rmtree(libs_dir)

        # Second pass: Install binary packages with Lambda platform
        self._logger.info("installing_binary_packages_for_lambda")
        binary_req_file = build_path / "binary_requirements.txt"
        binary_req_file.write_text("\n".join(binary_packages.keys()))

        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-r",
                str(binary_req_file),
                "-t",
                str(build_path),
                "--no-compile",
                "--quiet",
                "--platform",
                "manylinux2014_x86_64",
                "--implementation",
                "cp",
                "--python-version",
                "3.11",
                "--only-binary=:all:",
                "--no-deps",
            ],
            check=True,
            capture_output=True,
        )
        binary_req_file.unlink()

        # Clean up pip metadata
        for item in build_path.iterdir():
            if item.name.endswith(".dist-info") or item.name.endswith(".egg-info"):
                shutil.rmtree(item)

        # Remove __pycache__ directories
        for pycache in build_path.rglob("__pycache__"):
            shutil.rmtree(pycache)

    def _get_runtime_requirements(self) -> list[str]:
        """Get runtime requirements for Lambda."""
        # Core dependencies that DOVA needs at runtime
        # Exclude dev dependencies and packages that are too large
        runtime_deps = [
            "strands-agents>=0.1.0",
            "strands-agents-tools>=0.1.0",
            "bedrock-agentcore[strands-agents]>=1.0.6",
            "mcp>=1.21.0",
            "boto3>=1.35.0",
            "pydantic>=2.9.0",
            "pydantic-settings>=2.6.0",
            "python-dotenv>=1.0.0",
            "httpx>=0.28.0",
            "structlog>=24.4.0",
            "aiohttp>=3.9.0",
            "feedparser>=6.0.0",
            "beautifulsoup4>=4.12.0",
            "PyJWT>=2.8.0",
            "cryptography>=41.0.0",
            "tavily-python>=0.5.0",
        ]

        return runtime_deps

    def _create_bootstrap(self, build_path: Path) -> None:
        """Create Lambda handler bootstrap file."""
        self._logger.info("creating_bootstrap")

        bootstrap_content = '''"""Lambda bootstrap for DOVA.

This file is auto-generated by the DOVA Lambda packager.
It provides the entry point for AWS Lambda.
"""
from dova.runtime.lambda_handler import handler  # noqa: F401
'''
        (build_path / "lambda_function.py").write_text(bootstrap_content)

    def _calculate_size(self, path: Path) -> int:
        """Calculate total size of a directory."""
        total = 0
        for item in path.rglob("*"):
            if item.is_file():
                total += item.stat().st_size
        return total

    def _create_zip(self, source_dir: Path, zip_path: Path) -> None:
        """Create a ZIP file from a directory."""
        self._logger.info("creating_zip_file", path=str(zip_path))

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _dirs, files in os.walk(source_dir):
                for file in files:
                    file_path = Path(root) / file
                    arc_name = file_path.relative_to(source_dir)
                    zf.write(file_path, arc_name)
