# python main.py fit --model.model_head stp --model.rank 1 --model.horizon 1 --trainer.max_epochs 5
# python main.py fit --model.model_head cp --model.rank 2 --model.horizon 2 --trainer.max_epochs 5
python main.py fit --model.model_head cpme --model.rank 2 --model.horizon 2 --trainer.max_epochs 5
python main.py fit --model.model_head cp_cond --model.rank 2 --model.horizon 2 --trainer.max_epochs 5
python main.py fit --model.model_head cp_condl --model.rank 2 --model.horizon 2 --trainer.max_epochs 5
python main.py fit --model.model_head multihead --model.rank 2 --model.horizon 2 --trainer.max_epochs 5
