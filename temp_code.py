# Databricks notebook source
import torch
import gc
try:
  del llm
  torch.cuda.empty_cache()
  gc.collect()
except:
  print("test")
  pass

torch.cuda.empty_cache()
gc.collect()

# COMMAND ----------

from accelerate import init_empty_weights,infer_auto_device_map
from transformers import AutoConfig, AutoModelForCausalLM,TextStreamer
import transformers
from transformers import pipeline
import torch
from huggingface_hub import snapshot_download

name = "mosaicml/mpt-30b-chat"
# revision = "2abf1163dd8c9b11f07d805c06e6ec90a1f2037e"

snapshot_download(name,
                  # revision = revision,
                  resume_download=True,
                  local_files_only = False)

config_lm = AutoConfig.from_pretrained(name,
                                    # revision=revision,
                                    trust_remote_code=True)

config_lm.max_seq_len = 16384
config_lm.attn_config['attn_impl'] = 'triton'  # change this to use triton-based FlashAttention
config_lm.init_device = 'cuda:0'

with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config_lm,
                                             torch_dtype=torch.bfloat16,
                                             trust_remote_code=True)

model.tie_weights()

# COMMAND ----------

print(snapshot_download(name,
                  # revision = revision,
                  resume_download=True,
                  local_files_only = True))

# COMMAND ----------

# from accelerate import load_checkpoint_and_dispatch
# location = '/root/.cache/huggingface/hub/models--mosaicml--mpt-30b-chat/snapshots/7debc3fc2c5f330a33838bb007c24517b73347b8'
# model = load_checkpoint_and_dispatch(
#     model,location, device_map=device_map, no_split_module_classes=["MPTBlock"]
# )

model = transformers.AutoModelForCausalLM.from_pretrained(
  '/root/.cache/huggingface/hub/models--mosaicml--mpt-30b-chat/snapshots/7debc3fc2c5f330a33838bb007c24517b73347b8',
  config=config_lm,
  torch_dtype=torch.bfloat16, # Load model weights in bfloat16
  trust_remote_code=True,
  load_in_8bit=True,
  # revision=revision,
  device_map = 'auto'
)

# COMMAND ----------


from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('mosaicml/mpt-30b',
                                          padding='left'
                                          )
streamer = TextStreamer(tokenizer, skip_prompt=True)
generator = pipeline("text-generation",
                     model=model, 
                     config=config_lm, 
                     tokenizer=tokenizer,
                     streamer= streamer,
                     torch_dtype=torch.bfloat16)



def generate_text(prompt, **kwargs):
  if "max_new_tokens" not in kwargs:
    kwargs["max_new_tokens"] = 512
  
  kwargs.update(
        {
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
    )
  
  if "return_full_text" not in kwargs:
      kwargs["return_full_text"] = False

  
  template = "<|im_start|>system\n- You are a helpful assistant chatbot trained by MosaicML.\n- You answer questions.\n- You are more than just an information source,<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|><|im_start|>assistant\n"
  full_prompt = template.format(instruction=prompt)
  generated_text = generator(full_prompt, **kwargs)[0]["generated_text"]
  return generated_text


# COMMAND ----------

generate_text("""
screen, tyres and wheels, headlights, the undercarriage or the roof.\nb) loss of use of the rental vehicle due to damage.\nc) towing costs relating to damage.\nProvided that following an incident you are held responsible for the damage and are liable for an excess amount as specified in your rental agreement.\nPolicy limit – the most we will pay\nWe will reimburse you for the excess, up to a maximum of GBP 7,500 (or equivalent in local currency), under a single rental agreement. You can claim more than\nonce but in total we will only reimburse you up to a maximum of GBP 7,500 (or equivalent in local currency) during any one period of insurance. If a payment has\nbeen made in local currency any limits specified in this policy will be the equivalent in local currency based on the exchange rate that applied at the time of the\npurchase of your policy.\nN.B. Where you were covered by any other insurance for the same excess, we will only pay our share of the claim. At any point during the period of insurance we\nwill only cover one rental agreement, rental agreements may not overlap unless Family Cover has been selected and is applicable.\nAutomatic Extensions also included in the policy\nYour policy automatically includes cover for the following costs and services:\nMisfuelling Cover\nThis policy also covers you for costs incurred, up to a maximum of GBP 500 per claim, subject to a maximum of GBP 2,000\nin any one period of insurance, for cleaning out the engine and fuel system and any towing costs in the event that you\nput the wrong type of fuel in your rental vehicle.\nCar Rental Key Cover\nThis policy also covers you for costs incurred, up to a maximum of GBP 500 for each claim, subject to a maximum of GBP 2,000\nin any one period of insurance, for replacing a membership card/keys for a rental vehicle that is lost or stolen prior to the\nvehicle’s return, including replacement locks and locksmith charges.\n4\nPersonal Possessions Cover\nThis policy also covers you for costs incurred, up to a maximum of GBP 500, for your personal possessions damaged following attempted theft or stolen from the\nlocked boot or covered luggage area or glove box of the rental vehicle.\nThere is also a single article, pair or set limit of GBP 150. We will need an original proof of ownership or an insurance valuation in respect of all items claimed for. Where\nthese are not available the most we will pay is GBP 75 for each item, with a maximum of GBP 200 in total for all such items.\nAll claims for stolen personal possessions cover will require a crime reference number.\nPlease note that the policy does NOT cover the following:\n• Bonds, share certificates, guarantees or documents of any kind; or\n• Cash, traveller’s cheques or bank cards.\nSection II (only valid if ‘worldwide’ cover is selected and the additional premium is paid for ‘worldwide’ cover).\nCollision Damage Waiver (CDW)\nWe will indemnify you for losses incurred during a trip in or through USA and Canada including the Caribbean, South and Central America, as a result of damage\nto the rental vehicle following a covered incident up to the lowest of:\n• USD 100,000 (or the equivalent in local currency).\n• the value of the rental vehicle; or\n• the value of claim.\nWe will also pay legal costs incurred with our prior written consent for the defence of any claim which may be the subject of indemnity under this policy, subject to\nthe above limits.\nSection III (only valid if ‘SLI’ cover is selected and the additional premium for ‘SLI’ is paid)\nSupplemental Liability Insurance\nWe will indemnify you against all sums which you shall become legally liable to pay as damages and claimants’ costs in respect of bodily injury and damage to\nproperty arising out of an accident resulting from the use of a rental vehicle during the period of insurance for a trip in or through the USA, Canada, the Caribbean,\nSouth or Central America. The Indemnity provided by this policy shall apply only in excess of amounts recoverable under the primary liability insurance and the\nmaximum we will pay in respect of all claims arising from any one accident shall not exceed USD 1,000,000.\nThis Supplementary Liability Insurance Extension will not provide primary liability coverage and will only apply in excess of the primary liability insurance provided by\nthe car rental company or agency or primary liability insurance that has been sourced separately.\nCover provided by the car rental company or agency\nWhere liability insurance coverage is provided by the agreement between you and the car rental company or agency, the amount of such liability coverage may be\nadequate and supplementary liability coverage provided by this policy may not be required.\nSection IV \n based on the above paragraph and using no other information or context what is covered in case I lose my keys?""", temperature=0.1, max_new_tokens=256)

# COMMAND ----------

question =   "what is the start date of the policy mentioned in the  Policy Schedule / Validation Certificate?"
config['human_message_template'] = """<|im_start|>system\n-You are a helpful assistant chatbot trained by MosaicML.\n-You answer questions.-if you cannot find the answer , say I do not know. \n-If the query doesn't form a complete question, just say I don't know\n-If there is a good answer from the context, try to summarize the context to answer the question. \n<|im_end|>\n<|im_start|>user\n Given the context: {context}. Answer the question {question} <|im_end|><|im_start|>assistant\n""".strip()
