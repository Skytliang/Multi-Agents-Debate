set -e
set -u

MAD_PATH=$(realpath `dirname $0`)

python $MAD_PATH/code/debate4tran.py \
    -i $MAD_PATH/data/CommonMT/input.example.txt \
    -o $MAD_PATH/data/CommonMT/output \
    -lp zh-en \
    -k Your-OpenAI-Api-Key \
