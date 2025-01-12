# Environment variables for the project
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Add the project's src directory to PYTHONPATH dynamically
export PYTHONPATH=$PYTHONPATH:$(pwd)/src