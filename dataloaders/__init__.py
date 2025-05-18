# from dataloaders._base import BaseChatTemplate
from dataloaders.base import AbstractDataset
from dataloaders.gms8k import GSM8k
from dataloaders.stemp import STemp

# from dataloaders.stemp_v1 import ChatTemplateSynTemp, load_syn_temp_data


# CHAT_TEMPLATES: dict[str, type[BaseChatTemplate]] = {
#     "gsm8k": ChatTemplateGSM8k,
#     "stemp": ChatTemplateSynTemp,
# }


# DATASET_LOADERS = {
#     "gsm8k": load_gsm8k_data,
#     "stemp": load_syn_temp_data,
# }

DATASETS: dict[str, type[AbstractDataset]] = {
    "gsm8k": GSM8k,
    "stemp": STemp,
}
