import torch, os
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset, concatenate_datasets
from wechsel import WECHSEL, load_embeddings

import transformers

data_server = os.environ['AI_DATA_SERVER']
target_save = "/home/chang/AI/llm/t5tests/gpt-j/StockModels/llama-7B-ko-wechsel-v2"
llama_tokenizer_path = "/home/chang/AI/llm/t5tests/gpt-j/StockModels/llama-7B-origianal"

special_tokens = ["<|endoftext|>", "<unk>", "<cls>", "<sep>", "<mask>"]

# model = AutoModel.from_pretrained("chavinlo/alpaca-native", )
# model.save_pretrained("/home/chang/AI/llm/t5tests/gpt-j/StockModels/chavinlo-alpaca-native")
# exit()

# target_tokenizer = AutoTokenizer.from_pretrained(target_save)
# target_tokenizer.save_pretrained(target_save)
# exit()

from tokenizers import SentencePieceBPETokenizer

source_tokenizer = AutoTokenizer.from_pretrained(llama_tokenizer_path)
source_tokenizer.vocab = source_tokenizer.get_vocab()

target_tokenizer_base = SentencePieceBPETokenizer()
target_tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_object=target_tokenizer_base)

# polyglot_tokenizer = AutoTokenizer.from_pretrained("/home/chang/AI/llm/t5tests/gpt-j/Models/polyglot-ko-5.8b-odd-n-final-layers-tarot-10samp/final") 
wiki = load_dataset("lcw99/wikipedia-korean-20221001", split="train")
# namu = load_dataset("heegyu/namuwiki-extracted", split="train")
translation = load_dataset("json", data_files={'train': f"{data_server}aihub_translation.zip"})['train']
merged = concatenate_datasets([wiki])
target_tokenizer = target_tokenizer.train_new_from_iterator(
    merged["text"],
    vocab_size=len(source_tokenizer),
    new_special_tokens=special_tokens
)

target_tokenizer.eos_token = "<|endoftext|>"
target_tokenizer.eos_token_id = target_tokenizer(target_tokenizer.eos_token)['input_ids'][0]
target_tokenizer.pad_token = target_tokenizer.eos_token
target_tokenizer.pad_token_id = target_tokenizer.eos_token_id
target_tokenizer.unk_token = "<unk>"
target_tokenizer.unk_token_id = target_tokenizer(target_tokenizer.unk_token)['input_ids'][0]
target_tokenizer.cls_token = "<cls>"
target_tokenizer.cls_token_id = target_tokenizer(target_tokenizer.cls_token)['input_ids'][0]
target_tokenizer.sep_token = "<sep>"
target_tokenizer.sep_token_id = target_tokenizer(target_tokenizer.sep_token)['input_ids'][0]
target_tokenizer.mask_token = "<mask>"
target_tokenizer.mask_token_id = target_tokenizer(target_tokenizer.mask_token)['input_ids'][0]

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