#!/bin/bash
# Run comprehensive test suite with coverage

echo "=========================================="
echo "LLM Behaviour Lab - Test Suite"
echo "=========================================="
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Install test dependencies
echo "Installing test dependencies..."
pip install -q -r requirements-dev.txt

echo ""
echo "Running tests with coverage..."
echo ""

# Run pytest with coverage
pytest tests/ \
    --cov=. \
    --cov-report=html \
    --cov-report=term-missing \
    --cov-report=json \
    -v

# Check exit code
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ All tests passed!"
    echo "=========================================="
    echo ""
    echo "Coverage report generated:"
    echo "  - HTML: htmlcov/index.html"
    echo "  - JSON: coverage.json"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "✗ Some tests failed"
    echo "=========================================="
    echo ""
    exit 1
fi
