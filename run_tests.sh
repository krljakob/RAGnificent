#!/bin/bash
# Test runner script with different configurations

echo "RAGnificent Test Runner"
echo "====================="
echo ""

# Function to run tests with a specific configuration
run_tests() {
    local description=$1
    local args=$2
    
    echo "Running: $description"
    echo "Command: pytest $args"
    echo "-------------------"
    
    pytest $args
    
    echo ""
    echo "====================="
    echo ""
}

# Check command line arguments
if [ "$1" == "fast" ]; then
    echo "Running FAST tests only (no benchmarks, slow, or integration tests)"
    run_tests "Fast Unit Tests" "-m 'not benchmark and not slow and not integration and not requires_model' --tb=short"
    
elif [ "$1" == "unit" ]; then
    echo "Running UNIT tests only"
    run_tests "Unit Tests" "-m 'not integration' --tb=short"
    
elif [ "$1" == "integration" ]; then
    echo "Running INTEGRATION tests only"
    run_tests "Integration Tests" "-m integration"
    
elif [ "$1" == "benchmark" ]; then
    echo "Running BENCHMARK tests only"
    run_tests "Benchmark Tests" "-m benchmark"
    
elif [ "$1" == "all" ]; then
    echo "Running ALL tests (including slow ones)"
    run_tests "All Tests" "--tb=short"
    
elif [ "$1" == "profile" ]; then
    echo "Running tests with profiling"
    run_tests "Profile Tests" "--durations=20 -m 'not benchmark'"
    
else
    echo "Usage: $0 [fast|unit|integration|benchmark|all|profile]"
    echo ""
    echo "Options:"
    echo "  fast        - Run only fast unit tests (recommended for development)"
    echo "  unit        - Run all unit tests (excludes integration)"
    echo "  integration - Run only integration tests"
    echo "  benchmark   - Run only benchmark/performance tests"
    echo "  all         - Run all tests including slow ones"
    echo "  profile     - Run tests with duration profiling"
    echo ""
    echo "Default: Running fast tests..."
    echo ""
    run_tests "Fast Unit Tests" "-m 'not benchmark and not slow and not integration and not requires_model' --tb=short"
fi