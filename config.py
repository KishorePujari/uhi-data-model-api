import os 
import sys

my_path = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH="./models"

ENGLISH_VOCAB_FILE=os.path.join(MODEL_PATH,"english-word-list.txt")

WORD_FREQUENCIES_MODEL=os.path.join(MODEL_PATH,"word_frequencies.pickle")

NER_MODEL_PATH=os.path.join(MODEL_PATH,"ner_custom")


