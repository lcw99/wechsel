import torch, os
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset, concatenate_datasets
from wechsel import WECHSEL, load_embeddings

import transformers

data_server = os.environ['AI_DATA_SERVER']
target_save = "/home/chang/AI/llm/t5tests/gpt-j/StockModels/llama-7B-ko-wechsel-v3"
llama_tokenizer_path = "/home/chang/AI/llm/t5tests/gpt-j/StockModels/llama-7B-origianal"
target_tokenizer_path = "/home/chang/AI/llm/t5tests/gpt-j/StockModels/sentencepiece_wiki_kor_tokenizer"

special_tokens = ["<|endoftext|>", "<unk>", "<cls>", "<sep>", "<mask>"]

from tokenizers import SentencePieceBPETokenizer

source_tokenizer = AutoTokenizer.from_pretrained(llama_tokenizer_path)
source_tokenizer.vocab = source_tokenizer.get_vocab()

target_tokenizer = AutoTokenizer.from_pretrained(target_tokenizer_path)
target_tokenizer.vocab = target_tokenizer.get_vocab()

target_tokenizer.save_pretrained(target_save)

str = "당신은 소년입니다."
input_ids = target_tokenizer(str)['input_ids']
print(f"{len(input_ids)=}, {len(str)=}")

model = AutoModel.from_pretrained("decapoda-research/llama-7b-hf", )    # "chavinlo/alpaca-native"
wechsel = WECHSEL(
    load_embeddings("en"),
    load_embeddings("ko"),
    bilingual_dictionary="/home/chang/AI/llm/cc-kedict/korean_new.txt"
)

target_embeddings, info = wechsel.apply(
    source_tokenizer,
    target_tokenizer,
    model.get_input_embeddings().weight.detach().numpy(),
)

model.get_input_embeddings().weight.data = torch.from_numpy(target_embeddings)
model.save_pretrained(target_save)