set -e
set -u

MAD_PATH=$(realpath `dirname $0`)

# API key and model are read from the .env file (default: /workspace/BLINK/.env).
# Override the model with -m, or point to a different env file with --env-file.
python3 $MAD_PATH/code/debate4tran.py \
    -i $MAD_PATH/data/CommonMT/input.example.txt \
    -o $MAD_PATH/data/CommonMT/output \
    -lp zh-en \
    --env-file /workspace/BLINK/.env
