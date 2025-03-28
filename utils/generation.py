import token
from typing import Optional, Union

from typing import Optional
from sympy import true
import wandb

from transformers import TrainerCallback
from dataloaders.common import BaseChatTemplate
from tjdnet.models._tjd import TJD

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from utils.utils import truncate_tens


class GenerationCallback(TrainerCallback):
    def __init__(
        self,
        model: TJD,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        generate_strategy: str = "steps",
        generate_steps: int = 1000,
        max_new_tokens: int = 100,
        horizon: int = 1,
        chat_template: Optional[BaseChatTemplate] = None,
        top_k: int = 50,
        num_beams: int = 1,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.generate_strategy = generate_strategy
        self.generate_steps = generate_steps
        self.max_new_tokens = max_new_tokens
        self.horizon = horizon
        self.prompts = [chat_template.get_sample_prompt() if chat_template else ""]
        self.top_k = top_k
        self.num_beams = num_beams

    def on_step_end(self, args, state, control, **kwargs):
        if not args.local_rank == 0:
            return

        should_generate = False
        if self.generate_strategy == "steps":
            should_generate = state.global_step % self.generate_steps == 0
        elif self.generate_strategy == "epoch":
            # Check if we're at the end of an epoch
            steps_per_epoch = state.max_steps // state.num_train_epochs
            should_generate = state.global_step % steps_per_epoch == 0
        elif self.generate_strategy == "no":
            should_generate = False

        if should_generate:
            print("\n=== Generation Sample at step", state.global_step, "===")
            # Details
            print(f"global_step: {state.global_step}")
            print(f"num_train_epochs: {state.num_train_epochs}")
            self.model.eval()

            for i, prompt in enumerate(self.prompts):
                inputs = self.tokenizer(prompt, return_tensors="pt").input_ids.to(
                    self.model.device
                )
                outputs = self.model.generate(
                    input_ids=inputs,  # (batch_size, max_seq_len)
                    max_new_tokens=self.max_new_tokens,
                    top_k=self.top_k,
                    do_sample=True,
                    horizon=self.horizon,
                    stop_token=self.tokenizer.eos_token_id,  # type: ignore
                )  # (batch_size, max_seq_len') max_seq_len' might be less than max_seq_len if all sequences stopped early
                sample = self.tokenizer.decode(
                    truncate_tens(outputs[0], self.tokenizer.eos_token_id)  # type: ignore
                )
                print(f"\nPrompt:\n{prompt}\nOutput:\n{sample}\n")
                wandb.log(
                    {f"generation_text_{i}": wandb.Html(f"<pre>{sample}</pre>")},
                    step=state.global_step,
                )

            self.model.train()
            print("=" * 50 + "\n")
