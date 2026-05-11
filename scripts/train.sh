
cd ../

# IARD-RM (기본 설정 - sentence_transformer 사용)
python main.py --model iard_rm --dataset Amazon_Office_Products_14 --mode train
# NeuMF (리뷰 안 쓰는 모델)
python main.py --model neumf --dataset Amazon_Office_Products_14 --mode train
# RGCL (BERT-Whitening 사용)
python main.py --model rgcl --dataset Amazon_Office_Products_14 --mode train \
  --review_feature_backend bert_whitening \
  --bert_whitening_dim 64
# NARRE (기본 설정)
python main.py --model narre --dataset Amazon_Office_Products_14 --mode train
# DeepCoNN (기본 설정)
python main.py --model deepconn --dataset Amazon_Office_Products_14 --mode train
# SGDN (BERT-Whitening 사용)
python main.py --model sgdn --dataset Amazon_Office_Products_14 --mode train \
  --review_feature_backend bert_whitening \
  --bert_whitening_dim 64
# SSG (BERT-Whitening 사용 + embedding mode 필요)
python main.py --model ssg --dataset Amazon_Office_Products_14 --mode train \
  --review_feature_backend bert_whitening \
  --bert_whitening_dim 64 \
  --review_input_mode embedding