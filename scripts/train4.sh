
cd ../

# IARD-RM (기본 설정 - sentence_transformer 사용)
python main.py --model iard_rm --dataset Amazon_Office_Products_14 --mode train --gpu 2
# NeuMF (리뷰 안 쓰는 모델)
python main.py --model neumf --dataset Amazon_Office_Products_14 --mode train --gpu 2
python main.py --model narre --dataset Amazon_Office_Products_14 --mode train --gpu 2
# DeepCoNN (기본 설정)
python main.py --model deepconn --dataset Amazon_Office_Products_14 --mode train --gpu 2

# RGCL: 원본 ReviewGraph에 가깝게
python main.py --model rgcl --dataset Amazon_Office_Products_14 --mode train \
  --review_feature_backend bert_whitening \
  --bert_whitening_dim 64 --gpu 2

# SGDN: 원본 SGDN에 가깝게
python main.py --model sgdn --dataset Amazon_Office_Products_14 --mode train \
  --review_feature_backend bert_whitening \
  --bert_whitening_dim 64 --gpu 2

# SSG: 원본 SSG에 가깝게. BERT-Whitening/embedding mode 쓰지 않음.
python main.py --model ssg --dataset Amazon_Office_Products_14 --mode train \
  --review_input_mode token --gpu 2