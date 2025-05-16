from dataloaders._base import BaseChatTemplate
from dataloaders.gsm8k import ChatTemplateGSM8k, load_gsm8k_data
from dataloaders.syn_temp import ChatTemplateSynTemp, load_syn_temp_data


CHAT_TEMPLATES: dict[str, type[BaseChatTemplate]] = {
    "gsm8k": ChatTemplateGSM8k,
    "stemp": ChatTemplateSynTemp,
}


DATASET_LOADERS = {
    "gsm8k": load_gsm8k_data,
    "stemp": load_syn_temp_data,
}
