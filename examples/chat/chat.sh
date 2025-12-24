#!/bin/bash
# Quick launcher for Qwen2.5-VL model chat testing

# Default model path
MODEL_PATH="/default/hf/model/path"

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}Qwen2.5-VL Chat Tester${NC}"
echo -e "${BLUE}================================${NC}"
echo ""

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo -e "${YELLOW}Warning: Model not found at $MODEL_PATH${NC}"
    echo "Please update MODEL_PATH in this script or provide --model_path argument"
    echo ""
fi

# Parse mode argument
MODE="${1:-interactive}"

case "$MODE" in
    "interactive" | "-i")
        echo -e "${GREEN}Starting interactive mode...${NC}"
        python test_chat.py --model_path "$MODEL_PATH"
        ;;

    "batch" | "-b")
        if [ -z "$2" ]; then
            echo "Usage: $0 batch <json_file> [output_file]"
            exit 1
        fi
        BATCH_FILE="$2"
        OUTPUT_FILE="${3:-results.json}"
        echo -e "${GREEN}Starting batch mode...${NC}"
        echo "Input: $BATCH_FILE"
        echo "Output: $OUTPUT_FILE"
        python test_chat.py \
            --model_path "$MODEL_PATH" \
            --batch "$BATCH_FILE" \
            --output "$OUTPUT_FILE"
        ;;

    "greedy" | "-g")
        echo -e "${GREEN}Starting interactive mode with greedy decoding...${NC}"
        python test_chat.py \
            --model_path "$MODEL_PATH" \
            --temperature 0.0
        ;;

    "creative" | "-c")
        echo -e "${GREEN}Starting interactive mode with creative sampling...${NC}"
        python test_chat.py \
            --model_path "$MODEL_PATH" \
            --temperature 1.0 \
            --top_p 0.9
        ;;

    "help" | "-h" | "--help")
        echo "Usage: $0 [MODE] [OPTIONS]"
        echo ""
        echo "Modes:"
        echo "  interactive, -i    Start interactive chat (default)"
        echo "  batch, -b <file>   Run batch testing from JSON file"
        echo "  greedy, -g         Interactive with greedy decoding (temp=0)"
        echo "  creative, -c       Interactive with creative sampling (temp=1.0)"
        echo "  help, -h           Show this help message"
        echo ""
        echo "Examples:"
        echo "  $0                              # Interactive mode"
        echo "  $0 -i                           # Interactive mode (explicit)"
        echo "  $0 -b tests.json results.json  # Batch mode"
        echo "  $0 -g                           # Greedy decoding"
        echo "  $0 -c                           # Creative sampling"
        echo ""
        echo "For more advanced options, use:"
        echo "  python test_chat.py --help"
        ;;

    *)
        echo -e "${YELLOW}Unknown mode: $MODE${NC}"
        echo "Use '$0 help' to see available modes"
        exit 1
        ;;
esac
