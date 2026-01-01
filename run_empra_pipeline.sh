#!/bin/bash
#
# EMPRA End-to-End Pipeline Script
# 
# This script runs the complete EMPRA adversarial attack pipeline:
#   1. Generate adversarial sentences using EMPRA attack method
#   2. Construct adversarial documents by merging sentences with original documents
#
# Usage:
#   ./run_empra_pipeline.sh [OPTIONS]
#


set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" >&2
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" >&2
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" >&2
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

# Print usage information
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Required Arguments:
  --data-dir DIR              Directory containing queries and target documents
  --collection-dir DIR        Directory containing collection.tsv
  --dataset-name NAME         Dataset name (e.g., trecdl2020)
  --target-type TYPE          Target document type: 'easy' or 'hard'
  --output-dir DIR            Output directory for all results
  --relevance-model PATH      Path to neural ranking modeldirectory for relevance scoring
  --query-collection PATH     Path to query collection TSV file
  --doc-collection PATH       Path to document collection TSV file

Optional Arguments:
  --max-iterations N          Maximum attack iterations (default: 25)
  --epsilon FLOAT             Epsilon constraint for perturbations (default: 0.01)
  --alpha FLOAT               Step size for gradient updates (default: 0.1)
  --embedding-batch-size N    Batch size for embedding batch (default: 100)
  --num-workers N             Number of parallel workers (default: 3)
  --embedding-model MODEL     embedding model to use
  --model-tag TAG             Tag for model used in output filenames (default: S1)
  --coh-weight FLOAT          Weight for coherence score (default: 0.5)
  --rel-weight FLOAT          Weight for relevance score (default: 0.5)
  --batch-size N              Batch size for BERT model inference (default: 32)
  --num-labels N              Number of labels for relevance model (default: 1)
  --device DEVICE             Device for BERT model: 'cuda', 'cpu', or 'auto' (default: auto)
  --skip-step1                Skip adversarial sentence generation (step 1)
  --skip-step2                Skip adversarial document construction (step 2)
  --help                      Show this help message


EOF
    exit 1
}

# Default values
MAX_ITERATIONS=25
EPSILON=0.01
ALPHA=0.1
EMBEDDING_BATCH_SIZE=100
NUM_WORKERS=3
EMBEDDING_MODEL=""
MODEL_TAG="S1"
COH_WEIGHT=0.5
REL_WEIGHT=0.5
BATCH_SIZE=32
NUM_LABELS=1
DEVICE="auto"
SKIP_STEP1=false
SKIP_STEP2=false

# Parse command-line arguments
DATA_DIR=""
COLLECTION_DIR=""
DATASET_NAME=""
TARGET_TYPE=""
OUTPUT_DIR=""
RELEVANCE_MODEL=""
QUERY_COLLECTION=""
DOC_COLLECTION=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --collection-dir)
            COLLECTION_DIR="$2"
            shift 2
            ;;
        --dataset-name)
            DATASET_NAME="$2"
            shift 2
            ;;
        --target-type)
            TARGET_TYPE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --relevance-model)
            RELEVANCE_MODEL="$2"
            shift 2
            ;;
        --query-collection)
            QUERY_COLLECTION="$2"
            shift 2
            ;;
        --doc-collection)
            DOC_COLLECTION="$2"
            shift 2
            ;;
        --max-iterations)
            MAX_ITERATIONS="$2"
            shift 2
            ;;
        --epsilon)
            EPSILON="$2"
            shift 2
            ;;
        --alpha)
            ALPHA="$2"
            shift 2
            ;;
        --embedding-batch-size)
            EMBEDDING_BATCH_SIZE="$2"
            shift 2
            ;;
        --num-workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --embedding-model)
            EMBEDDING_MODEL="$2"
            shift 2
            ;;
        --api-key)
            API_KEY="$2"
            shift 2
            ;;
        --model-tag)
            MODEL_TAG="$2"
            shift 2
            ;;
        --coh-weight)
            COH_WEIGHT="$2"
            shift 2
            ;;
        --rel-weight)
            REL_WEIGHT="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --num-labels)
            NUM_LABELS="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --skip-step1)
            SKIP_STEP1=true
            shift
            ;;
        --skip-step2)
            SKIP_STEP2=true
            shift
            ;;
        --help)
            usage
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate required arguments
validate_required() {
    local missing=()
    
    [[ -z "$DATA_DIR" ]] && missing+=("--data-dir")
    [[ -z "$COLLECTION_DIR" ]] && missing+=("--collection-dir")
    [[ -z "$DATASET_NAME" ]] && missing+=("--dataset-name")
    [[ -z "$TARGET_TYPE" ]] && missing+=("--target-type")
    [[ -z "$OUTPUT_DIR" ]] && missing+=("--output-dir")
    
    if [[ "$SKIP_STEP1" == false ]]; then
        # Step 1 doesn't need relevance-model, query-collection, doc-collection
        :
    fi
    
    if [[ "$SKIP_STEP2" == false ]]; then
        [[ -z "$RELEVANCE_MODEL" ]] && missing+=("--relevance-model")
        [[ -z "$QUERY_COLLECTION" ]] && missing+=("--query-collection")
        [[ -z "$DOC_COLLECTION" ]] && missing+=("--doc-collection")
    fi
    
    if [[ ${#missing[@]} -gt 0 ]]; then
        log_error "Missing required arguments: ${missing[*]}"
        usage
    fi
}

# Validate file/directory existence
validate_paths() {
    local errors=0
    
    if [[ "$SKIP_STEP1" == false ]]; then
        [[ ! -d "$DATA_DIR" ]] && log_error "Data directory not found: $DATA_DIR" && errors=$((errors + 1))
        [[ ! -d "$COLLECTION_DIR" ]] && log_error "Collection directory not found: $COLLECTION_DIR" && errors=$((errors + 1))
        
        if [[ ! -f "$DATA_DIR/queries.tsv" ]]; then
            log_warning "queries.tsv not found in $DATA_DIR (expected but may be optional)"
        fi
        
        if [[ ! -f "$COLLECTION_DIR/collection.tsv" ]]; then
            log_error "collection.tsv not found in $COLLECTION_DIR"
            errors=$((errors + 1))
        fi
        
        if [[ ! -f "$DATA_DIR/rerank.${TARGET_TYPE}-5targets.tsv" ]]; then
            log_error "Target file not found: $DATA_DIR/rerank.${TARGET_TYPE}-5targets.tsv"
            errors=$((errors + 1))
        fi
    fi
    
    if [[ "$SKIP_STEP2" == false ]]; then
        [[ ! -d "$RELEVANCE_MODEL" ]] && log_error "Relevance model directory not found: $RELEVANCE_MODEL" && errors=$((errors + 1))
        [[ ! -f "$QUERY_COLLECTION" ]] && log_error "Query collection file not found: $QUERY_COLLECTION" && errors=$((errors + 1))
        [[ ! -f "$DOC_COLLECTION" ]] && log_error "Document collection file not found: $DOC_COLLECTION" && errors=$((errors + 1))
    fi
    
    if [[ $errors -gt 0 ]]; then
        log_error "Validation failed. Please fix the errors above."
        exit 1
    fi
}

# Validate target type
if [[ -n "$TARGET_TYPE" ]] && [[ "$TARGET_TYPE" != "easy" ]] && [[ "$TARGET_TYPE" != "hard" ]]; then
    log_error "Invalid target-type: $TARGET_TYPE (must be 'easy' or 'hard')"
    exit 1
fi

# Check for Python
if ! command -v python3 &> /dev/null; then
    log_error "python3 not found. Please install Python 3."
    exit 1
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$SCRIPT_DIR/code"

# Validate script files exist
if [[ "$SKIP_STEP1" == false ]]; then
    if [[ ! -f "$CODE_DIR/scripts/adversarial_text_generator.py" ]]; then
        log_error "Script not found: $CODE_DIR/scripts/adversarial_text_generator.py"
        exit 1
    fi
fi

if [[ "$SKIP_STEP2" == false ]]; then
    if [[ ! -f "$CODE_DIR/scripts/construct_adversarial_documents.py" ]]; then
        log_error "Script not found: $CODE_DIR/scripts/construct_adversarial_documents.py"
        exit 1
    fi
fi

# Main execution
main() {
    log_info "=========================================="
    log_info "EMPRA End-to-End Pipeline"
    log_info "=========================================="
    log_info "Dataset: $DATASET_NAME"
    log_info "Target Type: $TARGET_TYPE"
    log_info "Output Directory: $OUTPUT_DIR"
    log_info "=========================================="
    
    # Validate arguments
    validate_required
    validate_paths
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    log_success "Output directory created/verified: $OUTPUT_DIR"
    
    # Step 1: Generate adversarial sentences
    if [[ "$SKIP_STEP1" == false ]]; then
        log_info ""
        log_info "=========================================="
        log_info "Step 1: Generating Adversarial Sentences"
        log_info "=========================================="
        
        local step1_cmd=(
            python3 "$CODE_DIR/scripts/adversarial_text_generator.py"
            --data-dir "$DATA_DIR"
            --collection-dir "$COLLECTION_DIR"
            --dataset-name "$DATASET_NAME"
            --target-type "$TARGET_TYPE"
            --output-dir "$OUTPUT_DIR"
            --max-iterations "$MAX_ITERATIONS"
            --epsilon "$EPSILON"
            --alpha "$ALPHA"
            --embedding-batch-size "$EMBEDDING_BATCH_SIZE"
            --num-workers "$NUM_WORKERS"
            --embedding-model "$EMBEDDING_MODEL"
        )
        
        if [[ -n "$API_KEY" ]]; then
            step1_cmd+=(--api-key "$API_KEY")
        fi
        
        log_info "Running: ${step1_cmd[*]}"
        
        if "${step1_cmd[@]}"; then
            log_success "Step 1 completed successfully!"
        else
            log_error "Step 1 failed!"
            exit 1
        fi
    else
        log_warning "Skipping Step 1: Adversarial sentence generation"
    fi
    
    # Determine output file from step 1
    local adv_sentences_file="$OUTPUT_DIR/$DATASET_NAME/empra_generated_adversarial_sentences_${TARGET_TYPE}_docs.tsv"
    local target_file="$DATA_DIR/rerank.${TARGET_TYPE}-5targets.tsv"
    
    # Step 2: Construct adversarial documents
    if [[ "$SKIP_STEP2" == false ]]; then
        # Check if step 1 output exists (unless step 1 was skipped)
        if [[ "$SKIP_STEP1" == false ]] && [[ ! -f "$adv_sentences_file" ]]; then
            log_error "Adversarial sentences file not found: $adv_sentences_file"
            log_error "Step 1 may have failed or produced output in a different location."
            exit 1
        fi
        
        # If step 1 was skipped, allow user to specify the file
        if [[ "$SKIP_STEP1" == true ]] && [[ ! -f "$adv_sentences_file" ]]; then
            log_warning "Adversarial sentences file not found at expected location: $adv_sentences_file"
            log_warning "Please ensure the file exists or provide the correct path."
            read -p "Enter path to adversarial sentences file (or press Enter to continue anyway): " custom_file
            if [[ -n "$custom_file" ]] && [[ -f "$custom_file" ]]; then
                adv_sentences_file="$custom_file"
            elif [[ ! -f "$adv_sentences_file" ]]; then
                log_error "Cannot proceed without adversarial sentences file."
                exit 1
            fi
        fi
        
        log_info ""
        log_info "=========================================="
        log_info "Step 2: Constructing Adversarial Documents"
        log_info "=========================================="
        
        local step2_cmd=(
            python3 "$CODE_DIR/scripts/construct_adversarial_documents.py"
            --relevance-model "$RELEVANCE_MODEL"
            --model-tag "$MODEL_TAG"
            --connect-sent-file "$adv_sentences_file"
            --target-file "$target_file"
            --query-collection "$QUERY_COLLECTION"
            --doc-collection "$DOC_COLLECTION"
            --coh-weight "$COH_WEIGHT"
            --rel-weight "$REL_WEIGHT"
            --batch-size "$BATCH_SIZE"
            --num-labels "$NUM_LABELS"
        )
        
        if [[ "$DEVICE" != "auto" ]]; then
            step2_cmd+=(--device "$DEVICE")
        fi
        
        log_info "Running: ${step2_cmd[*]}"
        
        if "${step2_cmd[@]}"; then
            log_success "Step 2 completed successfully!"
        else
            log_error "Step 2 failed!"
            exit 1
        fi
    else
        log_warning "Skipping Step 2: Adversarial document construction"
    fi
    
    log_info ""
    log_info "=========================================="
    log_success "Pipeline completed successfully!"
    log_info "=========================================="
    log_info "Results are available in: $OUTPUT_DIR"
    
    if [[ "$SKIP_STEP1" == false ]]; then
        log_info "Adversarial sentences: $adv_sentences_file"
    fi
    
    if [[ "$SKIP_STEP2" == false ]]; then
        log_info "Adversarial documents should be in the output directory"
    fi
}

# Run main function
main "$@"

