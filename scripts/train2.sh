
cd ../

# IARD-RM (기본 설정 - sentence_transformer 사용)
python main.py --model iard_rm --dataset Amazon_Toys_and_Games_14 --mode train
# NeuMF (리뷰 안 쓰는 모델)
python main.py --model neumf --dataset Amazon_Toys_and_Games_14 --mode train
python main.py --model narre --dataset Amazon_Toys_and_Games_14 --mode train
# DeepCoNN (기본 설정)
python main.py --model deepconn --dataset Amazon_Toys_and_Games_14 --mode train

# RGCL: 원본 ReviewGraph에 가깝게
python main.py --model rgcl --dataset Amazon_Toys_and_Games_14 --mode train

# SGDN: 원본 SGDN에 가깝게
python main.py --model sgdn --dataset Amazon_Toys_and_Games_14 --mode train

# SSG: 원본 SSG에 가깝게. BERT-Whitening/embedding mode 쓰지 않음.
python main.py --model ssg --dataset Amazon_Toys_and_Games_14 --mode train
