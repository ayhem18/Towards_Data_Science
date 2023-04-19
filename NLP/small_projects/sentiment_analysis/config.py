from transformers import BertTokenizer 


MAX_LENGTH = 512
TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 4
EPOCHS = 10
ACCUMULATION = 2
BERT_PATH = 'best_base_cased'
TOKENIZER = BertTokenizer.from_pretrained(BERT_PATH)

BERT_OUTPUT_UNITS = 768
