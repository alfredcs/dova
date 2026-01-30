#!/usr/bin/env bash
#
# DOVA Development Environment Setup Script
#
# This script sets up the local development environment for DOVA.
# It installs Python dependencies, sets up pre-commit hooks,
# and configures local services.
#

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Log functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check Python version
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is not installed. Please install Python 3.11+."
        exit 1
    fi

    PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    REQUIRED_VERSION="3.11"

    if [[ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]]; then
        log_error "Python $REQUIRED_VERSION+ is required. Found: $PYTHON_VERSION"
        exit 1
    fi

    log_success "Python $PYTHON_VERSION found"

    # Check uv (preferred) or pip
    if command -v uv &> /dev/null; then
        log_success "uv package manager found"
        PACKAGE_MANAGER="uv"
    elif command -v pip &> /dev/null; then
        log_warning "uv not found, falling back to pip"
        PACKAGE_MANAGER="pip"
    else
        log_error "Neither uv nor pip found. Please install uv: curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi

    # Check Docker (optional)
    if command -v docker &> /dev/null; then
        log_success "Docker found"
    else
        log_warning "Docker not found. Local services will need to be installed manually."
    fi

    # Check AWS CLI (optional)
    if command -v aws &> /dev/null; then
        log_success "AWS CLI found"
    else
        log_warning "AWS CLI not found. Required for deployment."
    fi
}

# Create virtual environment
setup_venv() {
    log_info "Setting up Python virtual environment..."

    cd "$PROJECT_ROOT"

    if [[ "$PACKAGE_MANAGER" == "uv" ]]; then
        uv venv .venv
        source .venv/bin/activate
        uv pip install -e ".[dev]"
    else
        python3 -m venv .venv
        source .venv/bin/activate
        pip install --upgrade pip
        pip install -e ".[dev]"
    fi

    log_success "Virtual environment created and dependencies installed"
}

# Setup environment variables
setup_env() {
    log_info "Setting up environment variables..."

    cd "$PROJECT_ROOT"

    if [[ ! -f .env ]]; then
        if [[ -f .env.example ]]; then
            cp .env.example .env
            log_success "Created .env from .env.example"
            log_warning "Please update .env with your actual configuration values"
        else
            log_warning "No .env.example found. Creating empty .env file"
            touch .env
        fi
    else
        log_info ".env file already exists, skipping"
    fi
}

# Setup pre-commit hooks
setup_hooks() {
    log_info "Setting up pre-commit hooks..."

    cd "$PROJECT_ROOT"

    if command -v pre-commit &> /dev/null; then
        pre-commit install
        log_success "Pre-commit hooks installed"
    else
        log_warning "pre-commit not found. Install with: pip install pre-commit"
    fi
}

# Setup local services with Docker
setup_services() {
    log_info "Setting up local services..."

    cd "$PROJECT_ROOT"

    if command -v docker &> /dev/null && command -v docker-compose &> /dev/null; then
        log_info "Starting Docker services..."
        docker-compose up -d redis
        log_success "Local services started"
    else
        log_warning "Docker/docker-compose not available. Skipping local services."
        log_info "You can start services manually or use managed services."
    fi
}

# Setup CDK infrastructure (optional)
setup_infra() {
    log_info "Setting up CDK infrastructure..."

    cd "$PROJECT_ROOT/infra"

    if command -v npm &> /dev/null; then
        npm install
        log_success "CDK dependencies installed"
    else
        log_warning "npm not found. CDK deployment will not be available."
    fi
}

# Verify installation
verify_installation() {
    log_info "Verifying installation..."

    cd "$PROJECT_ROOT"

    # Check if dova package is importable
    if python3 -c "import dova; print(f'DOVA version: {dova.__version__}')" 2>/dev/null; then
        log_success "DOVA package installed correctly"
    else
        log_error "Failed to import DOVA package"
        exit 1
    fi

    # Run a quick test
    if python3 -m pytest tests/unit/agents/test_orchestrator.py -v --tb=short 2>/dev/null; then
        log_success "Basic tests pass"
    else
        log_warning "Some tests failed. This may be expected for initial setup."
    fi
}

# Print next steps
print_next_steps() {
    echo ""
    echo "============================================"
    echo -e "${GREEN}DOVA Setup Complete!${NC}"
    echo "============================================"
    echo ""
    echo "Next steps:"
    echo "1. Activate the virtual environment:"
    echo "   source .venv/bin/activate"
    echo ""
    echo "2. Update .env with your configuration:"
    echo "   - AWS credentials"
    echo "   - LLM provider API keys"
    echo "   - MCP server configurations"
    echo ""
    echo "3. Run the development server:"
    echo "   make run-local"
    echo ""
    echo "4. Run tests:"
    echo "   make test"
    echo ""
    echo "5. Deploy to AWS:"
    echo "   make deploy"
    echo ""
    echo "For more information, see README.md"
    echo ""
}

# Main function
main() {
    echo ""
    echo "============================================"
    echo "DOVA Development Environment Setup"
    echo "============================================"
    echo ""

    check_prerequisites
    setup_venv
    setup_env
    setup_hooks
    setup_services
    setup_infra
    verify_installation
    print_next_steps
}

# Run main function
main "$@"
