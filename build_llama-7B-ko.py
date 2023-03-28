import torch
from transformers import AutoModel, AutoTokenizer, LlamaTokenizer
from datasets import load_dataset
from wechsel import WECHSEL, load_embeddings

source_tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
source_tokenizer.vocab = source_tokenizer.get_vocab()
model = AutoModel.from_pretrained("decapoda-research/llama-7b-hf")

# target_tokenizer = source_tokenizer.train_new_from_iterator(
#     load_dataset("oscar", "unshuffled_deduplicated_sw", split="train")["text"],
#     vocab_size=len(source_tokenizer)
# )

target_tokenizer = AutoTokenizer.from_pretrained("/home/chang/AI/llm/t5tests/gpt-j/Models/polyglot-ko-5.8b-odd-n-final-layers-tarot-10samp/final") 

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
model.save_pretrained("./llama-7B-ko-wechsel")