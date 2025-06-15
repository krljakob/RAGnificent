#!/bin/bash
# Profile the markdown converter with flamegraph

set -e

# Ensure we're in the project root
cd "$(dirname "$0")/../.."

# Build with debug symbols
echo "Building with debug symbols..."
cargo build --profile release-with-debug

# Check if we can run perf without sudo
if [ -w /proc/sys/kernel/perf_event_paranoid ]; then
    # We can write to perf_event_paranoid, no sudo needed
    PERF_CMD=""
else
    # Need sudo for perf
    PERF_CMD="sudo -E"
    echo "Note: Using sudo for perf access. You may need to enter your password."
fi

# Run flamegraph on the markdown converter benchmark
echo "Generating flamegraph..."
$PERF_CMD cargo flamegraph \
    --bench RAGnificent_bench \
    --output=flamegraph_markdown.svg \
    --deterministic \
    -- \
    --bench \
    --profile-time=5

if [ $? -eq 0 ]; then
    echo "Flamegraph generated at $(pwd)/flamegraph_markdown.svg"
    
    # Open the flamegraph in the default browser if possible
    if command -v xdg-open > /dev/null; then
        xdg-open flamegraph_markdown.svg 2>/dev/null
    elif command -v open > /dev/null; then
        open flamegraph_markdown.svg 2>/dev/null
    fi
else
    echo "Failed to generate flamegraph. You may need to run with sudo or adjust perf permissions."
    echo "Try running: sudo sh -c 'echo -1 > /proc/sys/kernel/perf_event_paranoid'"
    exit 1
fi
