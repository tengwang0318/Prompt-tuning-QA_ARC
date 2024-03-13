### Easy

test max_len 128

```
python --data_path "data/ARC-Easy-validation.jsonl" --device_id "0,1" --model "models/phi-1_5" --embedder models/bge-small-en-v1.5 --start_index 0 --end_index 999999 --max_len 100 --output_path "validation_easy_outputs/" --overwrite True --prompt_type "v2.0" --N 8 --top_k True --top_k_reverse False
```

```
tokenizer.decode(input_ids)
'Question: Which best describes transportation technology?\nAnswer: a system that is used to move people and products\n\nQuestion: Which did Thomas Edison invent?\nAnswer: light bulb\n\nQuestion: Improvements in farming technology would most likely\nAnswer: increase the amount of food produced.\n\nQuestion: What is the first step in designing a product?\nAnswer: identify the need or want\n\nQuestion: A push or a pull on an object is an example of\nAnswer: force.'
source
'Question: Which best describes transportation technology?\nAnswer: a system that is used to move people and products\n\nQuestion: Which did Thomas Edison invent?\nAnswer: light bulb\n\nQuestion: Improvements in farming technology would most likely\nAnswer: increase the amount of food produced.\n\nQuestion: What is the first step in designing a product?\nAnswer: identify the need or want\n\nQuestion: A push or a pull on an object is an example of\nAnswer: force.\n\nQuestion: Which of the following is the best example of a custom-made product?\nAnswer: artificial leg\n\nQuestion: Which is the first step in a design process?\nAnswer: Describe the problem.\n\nQuestion: Which invention made mass production possible?\nAnswer: the assembly line\n\nQuestion: Which technology was developed most recently?\nAnswer:'

```

