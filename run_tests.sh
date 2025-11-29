#!/bin/bash

echo "================================"
echo "Running Neural Network Tests"
echo "================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Compile and run each test
tests=("test_matrix" "test_dense" "test_activation" "test_xor")
passed=0
failed=0

for test in "${tests[@]}"; do
    echo "Running $test..."
    if [ -f "build/$test" ] || [ -f "build/$test.exe" ]; then
        if ./build/$test 2>&1 || ./build/$test.exe 2>&1; then
            echo -e "${GREEN}✓ $test passed${NC}"
            ((passed++))
        else
            echo -e "${RED}✗ $test failed${NC}"
            ((failed++))
        fi
    else
        echo -e "${RED}✗ $test not found (not compiled)${NC}"
        ((failed++))
    fi
    echo ""
done

echo "================================"
echo "Test Summary"
echo "================================"
echo -e "Passed: ${GREEN}$passed${NC}"
echo -e "Failed: ${RED}$failed${NC}"
echo "================================"

if [ $failed -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed.${NC}"
    exit 1
fi
