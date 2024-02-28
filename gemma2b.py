#import pandas to work with tables
import pandas as pd
from datasets import load_dataset, Dataset
import json
from trl import SFTTrainer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
from trl import SFTTrainer
from transformers import TrainingArguments
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
import transformers
import wandb


df = pd.read_excel('finalData.xlsx') #add your filename or datatset file, this reads the file and store as table
df.head(5)
data_json = df.to_json(orient="records", indent=0)#converting the table into json
json_dataset = json.loads(data_json)#loads as json datatset
json_dataset

training_data = []
for i in json_dataset:#this converts the json into ### Human: .... ### Assitant: .... prompt style to train the model
    # Check if 'Prompt' or 'Response' is None and replace it with an empty string
    prompt = i.get('Prompt ', '') or ''
    response = i.get('Response ', '') or ''
    # Concatenate the strings
    prompt_str = prompt
    response_str = response
    training_data.append({"prompt": prompt_str, "response": response_str})
training_data
train_dataset = Dataset.from_list(training_data)#the list is converted into dataset using Dataset library
train_dataset=train_dataset.shuffle(seed=1234)


model_id = "mistralai/Mistral-7B-v0.1"#Here we should give the model name as it is in hugging face


tokenizer = AutoTokenizer.from_pretrained(model_id)#We get the respective tokenizer from huggging face using the model id

tokenizer.pad_token = tokenizer.unk_token
tokenizer.padding_side= "right"

model = AutoModelForCausalLM.from_pretrained(#loading the model from hugging face
    model_id,  # Llama 2 7b, same as before# Same quantization config as before
    device_map={"":0},
    trust_remote_code=True,
   # quantization_config=bnb_config
)

model.gradient_checkpointing_enable()#enabling this to store cehckpoints
model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(#This is an additional layer which will be linked to the model and we will train this weights
    r=128, #increase in this will lead to overfitting abd more training resources
    lora_alpha=256, #same for this
    lora_dropout=0.05, 
    bias="none", 
    task_type="CAUSAL_LM", 
    target_modules=["q_proj", "v_proj", "k_proj","o_proj","gate_proj", "up_proj", "down_proj","lm_head"]#module names we want to effect
)

model = get_peft_model(model, peft_config)#loading the model with lora layer

def formatting_func(example):
    text = f"<s>[INST] {example['prompt']} [/INST] {example['response']}</s>"
    return text

max_length=1024#max token size

def generate_and_tokenize_prompt2(prompt):#will tokenize the dataset
    result = tokenizer(
        formatting_func(prompt),
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result

def print_trainable_parameters(model):#checking how much parameters this effects
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    ) 

print_trainable_parameters(model)

tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt2)#this will tokenize the dataset

def formatting_prompts_func(x):
    output_texts = []
    for i in range(len(x['prompt'])):
        text = f"<s>[INST] {x['prompt'][i]}[/INST] {x['response'][i]}</s>"
        output_texts.append(text)
    return output_texts
#pip uninstall bitsandbytes


import transformers
from datetime import datetime
import bitsandbytes as bnb
project = "askme-govtech"#folder name of your choice
base_model_name = "mistralfinal"#model name of your choice
run_name = base_model_name + "-" + project
output_dir = "./" + run_name
trainer = transformers.Trainer(
    model=model,
    train_dataset=tokenized_train_dataset,
    args=transformers.TrainingArguments(
        output_dir=output_dir,
        warmup_steps=16,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        optim = "paged_adamw_32bit",
        gradient_checkpointing=True,
        max_steps=5000,
        learning_rate=2.5e-5, # Want a small lr for finetuning
        bf16=True,
        fp16=False,
        logging_steps=100,              # When to start reporting loss
        logging_dir="./logs",        # Directory for storing logs
        save_strategy="steps",       # Save the model checkpoint every logging step
        save_steps=25,                # Save checkpoints every 50 steps
        #report_to="wandb",           # Comment this out if you don't want to use weights & baises
        #run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"          # Name of the W&B run (optional)
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
# 'adamw_bnb_8bit', 'adamw_8bit', 'lion_8bit'

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()

#TrainOutput(global_step=500, training_loss=0.07163945285881346, metrics={'train_runtime': 17272.9291, 'train_samples_per_second': 0.926, 'train_steps_per_second': 0.029, 'train_loss': 0.07163945285881346, 'epoch': 14.39})
#Inferencing

base_model_id = "mistralai/Mistral-7B-v0.1"

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,  # Llama 2 7b, same as before# Same quantization config as before
    device_map="auto",
    trust_remote_code=True
)
base_model.config.use_cache = True
eval_tokenizer = AutoTokenizer.from_pretrained(base_model_id, add_bos_token=True)
eval_tokenizer.pad_token = eval_tokenizer.unk_token
eval_tokenizer.padding_side= "right"


from peft import PeftModel

ft_model = PeftModel.from_pretrained(base_model, "mistral3-askme-govtech/checkpoint-500")

eval_prompt = "<s>[INST] ####Human: Who is the secretary of GovTech Bhutan?[/INST]"
model_input = eval_tokenizer(eval_prompt, return_tensors="pt").to("cuda")

ft_model.eval()
with torch.no_grad():
    print(eval_tokenizer.decode(ft_model.generate(**model_input, max_new_tokens=256, repetition_penalty=1.15)[0], skip_special_tokens=True ))



##saving the merged model
merged_model = ft_model.merge_and_unload()
merged_model.save_pretrained("ask_me2")
eval_tokenizer.save_pretrained("ask_me2")
'''
ft_model.push_to_hub("Taphu/chatbot_mistral7b")
eval_tokenizer.push_to_hub("Taphu/chatbot_mistral7b")
'''
'''
python llama.cpp/convert.py foldername\
  --outfile output.gguf \
  --outtype q8_0
'''

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import LlamaCpp

llm = LlamaCpp(
    model_path="mistral7b_govtech_v2.gguf",
    n_gpu_layers=1,
    n_batch=512,
    n_ctx=2048,
    f16_kv=True,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    verbose=True,
)

llm("###Question: Who is the acting secretary of GovTech")

llm.invoke("###Question: Name the head of each division of GovTech agency of Bhutan")

