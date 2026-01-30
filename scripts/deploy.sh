#!/usr/bin/env bash
#
# DOVA Deployment Script
#
# This script deploys the DOVA platform to AWS using CDK.
# It builds the Docker image, pushes to ECR, and deploys infrastructure.
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

# Default values
ENVIRONMENT="${ENVIRONMENT:-dev}"
AWS_REGION="${AWS_REGION:-us-east-1}"
STACK_NAME="dova-${ENVIRONMENT}"

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

# Print usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -e, --environment    Environment to deploy (dev, staging, prod). Default: dev"
    echo "  -r, --region         AWS region. Default: us-east-1"
    echo "  -d, --dry-run        Perform a dry run (CDK diff only)"
    echo "  -h, --help           Show this help message"
    echo ""
    echo "Environment variables:"
    echo "  AWS_PROFILE          AWS profile to use"
    echo "  AWS_REGION           AWS region (overridden by --region)"
    echo "  ENVIRONMENT          Environment (overridden by --environment)"
    echo ""
}

# Parse arguments
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -r|--region)
            AWS_REGION="$2"
            shift 2
            ;;
        -d|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate environment
validate_environment() {
    case $ENVIRONMENT in
        dev|staging|prod)
            log_info "Deploying to environment: $ENVIRONMENT"
            ;;
        *)
            log_error "Invalid environment: $ENVIRONMENT. Must be dev, staging, or prod."
            exit 1
            ;;
    esac
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check AWS CLI
    if ! command -v aws &> /dev/null; then
        log_error "AWS CLI is not installed. Please install it first."
        exit 1
    fi

    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        log_error "AWS credentials not configured or expired."
        exit 1
    fi

    AWS_ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
    log_success "AWS Account: $AWS_ACCOUNT"

    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install it first."
        exit 1
    fi

    # Check CDK
    if ! command -v cdk &> /dev/null; then
        log_warning "CDK CLI not found globally, using npx"
        CDK_CMD="npx cdk"
    else
        CDK_CMD="cdk"
    fi

    # Check Node.js
    if ! command -v node &> /dev/null; then
        log_error "Node.js is not installed. Please install it first."
        exit 1
    fi

    log_success "All prerequisites met"
}

# Run tests before deployment
run_tests() {
    log_info "Running tests..."

    cd "$PROJECT_ROOT"

    if [[ -f .venv/bin/activate ]]; then
        source .venv/bin/activate
    fi

    # Run unit tests
    if python3 -m pytest tests/unit -v --tb=short; then
        log_success "Unit tests passed"
    else
        log_error "Unit tests failed. Aborting deployment."
        exit 1
    fi
}

# Build Docker image
build_image() {
    log_info "Building Docker image..."

    cd "$PROJECT_ROOT"

    # Get ECR repository URL
    ECR_REPO="${AWS_ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com/dova-${ENVIRONMENT}"

    # Login to ECR
    aws ecr get-login-password --region "$AWS_REGION" | \
        docker login --username AWS --password-stdin "${AWS_ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com"

    # Build image
    docker build \
        --platform linux/amd64 \
        --build-arg ENVIRONMENT="$ENVIRONMENT" \
        -t "dova:${ENVIRONMENT}" \
        -t "${ECR_REPO}:latest" \
        -t "${ECR_REPO}:$(git rev-parse --short HEAD)" \
        .

    log_success "Docker image built"
}

# Push image to ECR
push_image() {
    log_info "Pushing Docker image to ECR..."

    ECR_REPO="${AWS_ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com/dova-${ENVIRONMENT}"

    # Create ECR repository if it doesn't exist
    aws ecr describe-repositories --repository-names "dova-${ENVIRONMENT}" --region "$AWS_REGION" &> /dev/null || \
        aws ecr create-repository --repository-name "dova-${ENVIRONMENT}" --region "$AWS_REGION"

    # Push images
    docker push "${ECR_REPO}:latest"
    docker push "${ECR_REPO}:$(git rev-parse --short HEAD)"

    log_success "Docker image pushed to ECR"
}

# Deploy CDK stack
deploy_cdk() {
    log_info "Deploying CDK stack..."

    cd "$PROJECT_ROOT/infra"

    # Install dependencies if needed
    if [[ ! -d node_modules ]]; then
        npm install
    fi

    # Set environment variables for CDK
    export CDK_DEFAULT_ACCOUNT="$AWS_ACCOUNT"
    export CDK_DEFAULT_REGION="$AWS_REGION"
    export DOVA_ENVIRONMENT="$ENVIRONMENT"

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "Performing dry run (diff only)..."
        $CDK_CMD diff "$STACK_NAME" --context environment="$ENVIRONMENT"
    else
        # Bootstrap CDK if needed
        $CDK_CMD bootstrap "aws://${AWS_ACCOUNT}/${AWS_REGION}" || true

        # Deploy stack
        $CDK_CMD deploy "$STACK_NAME" \
            --context environment="$ENVIRONMENT" \
            --require-approval never \
            --outputs-file cdk-outputs.json

        log_success "CDK stack deployed"
    fi
}

# Print deployment info
print_deployment_info() {
    log_info "Deployment complete!"

    cd "$PROJECT_ROOT/infra"

    if [[ -f cdk-outputs.json ]]; then
        echo ""
        echo "============================================"
        echo "Deployment Outputs"
        echo "============================================"
        cat cdk-outputs.json
        echo ""
    fi

    echo ""
    echo "============================================"
    echo -e "${GREEN}DOVA Deployment Complete!${NC}"
    echo "============================================"
    echo ""
    echo "Environment: $ENVIRONMENT"
    echo "Region: $AWS_REGION"
    echo "Stack: $STACK_NAME"
    echo ""
    echo "Next steps:"
    echo "1. Verify the API endpoint in the CloudFormation outputs"
    echo "2. Test the health endpoint: curl <API_URL>/health"
    echo "3. Monitor logs in CloudWatch"
    echo ""
}

# Main function
main() {
    echo ""
    echo "============================================"
    echo "DOVA Deployment Script"
    echo "============================================"
    echo ""

    validate_environment
    check_prerequisites

    if [[ "$ENVIRONMENT" == "prod" ]]; then
        log_warning "Deploying to PRODUCTION environment!"
        read -p "Are you sure you want to continue? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Deployment cancelled"
            exit 0
        fi
        run_tests
    fi

    if [[ "$DRY_RUN" == "false" ]]; then
        build_image
        push_image
    fi

    deploy_cdk

    if [[ "$DRY_RUN" == "false" ]]; then
        print_deployment_info
    fi
}

# Run main function
main "$@"
