#!/bin/bash
# Run tests with different categories for the RAGnificent project

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Default mode
MODE="fast"

# Parse command line arguments
if [ $# -gt 0 ]; then
    MODE=$1
fi

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo -e "${YELLOW}Running tests in ${MODE} mode...${NC}"

case $MODE in
    "fast")
        echo -e "${GREEN}Running fast unit tests only (no benchmarks, no slow tests)${NC}"
        pytest -v -m "not benchmark and not slow and not integration and not requires_model" --tb=short
        ;;
    
    "unit")
        echo -e "${GREEN}Running all unit tests${NC}"
        pytest -v -m "unit or (not integration and not benchmark)" --tb=short
        ;;
    
    "integration")
        echo -e "${GREEN}Running integration tests${NC}"
        pytest -v -m "integration" --tb=short
        ;;
    
    "benchmark")
        echo -e "${GREEN}Running benchmark tests${NC}"
        pytest -v -m "benchmark" --tb=short
        ;;
    
    "all")
        echo -e "${GREEN}Running all tests including benchmarks${NC}"
        pytest -v --tb=short
        ;;
    
    "profile")
        echo -e "${GREEN}Running tests with duration profiling${NC}"
        pytest -v --durations=20 -m "not benchmark" --tb=short
        ;;
    
    "coverage")
        echo -e "${GREEN}Running tests with coverage report${NC}"
        pytest -v --cov=RAGnificent --cov-report=html --cov-report=term -m "not benchmark" --tb=short
        echo -e "${GREEN}Coverage report generated in htmlcov/index.html${NC}"
        ;;
    
    "rust")
        echo -e "${GREEN}Running Rust tests${NC}"
        cargo test --quiet
        ;;
    
    "all-with-rust")
        echo -e "${GREEN}Running all Python and Rust tests${NC}"
        pytest -v --tb=short
        cargo test --quiet
        ;;
    
    *)
        echo -e "${RED}Unknown mode: $MODE${NC}"
        echo "Available modes:"
        echo "  fast         - Run only fast unit tests (default)"
        echo "  unit         - Run all unit tests"
        echo "  integration  - Run integration tests"
        echo "  benchmark    - Run benchmark tests"
        echo "  all          - Run all Python tests"
        echo "  profile      - Run tests with duration profiling"
        echo "  coverage     - Run tests with coverage report"
        echo "  rust         - Run Rust tests only"
        echo "  all-with-rust - Run all Python and Rust tests"
        exit 1
        ;;
esac

# Check exit status
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Tests passed!${NC}"
else
    echo -e "${RED}❌ Tests failed!${NC}"
    exit 1
fi