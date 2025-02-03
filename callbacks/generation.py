from typing import Optional

from typing import Optional
import wandb

from transformers import TrainerCallback
from helpers import get_test_samples
from data.common import BaseChatTemplate


# TODO: dont need to pass model and tokenizer in init
class GenerationCallback(TrainerCallback):
    def __init__(
        self,
        model,
        tokenizer,
        generate_strategy="steps",
        generate_steps=1000,
        max_new_tokens=100,
        horizon=1,
        chat_template: Optional[BaseChatTemplate] = None,
        top_k=50,
        num_beams=1,
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
            should_generate = state.global_step % state.num_train_epochs == 0
        elif self.generate_strategy == "no":
            should_generate = False

        if should_generate:
            print("\n=== Generation Sample at step", state.global_step, "===")
            self.model.eval()

            samples = {}
            for i, prompt in enumerate(self.prompts):
                sample = get_test_samples(
                    self.model,
                    self.tokenizer,
                    prompt=prompt,
                    max_new_tokens=self.max_new_tokens,
                    horizon=self.horizon,
                    top_k=self.top_k,
                    num_beams=self.num_beams,
                )
                samples[f"prompt_{i+1}"] = prompt
                samples[f"generation_{i+1}"] = sample
                print(f"\nPrompt: {prompt}\nOutput: {sample}\n")
                wandb.log(
                    {f"generation_text_{i}": wandb.Html(f"<pre>{sample}</pre>")},
                    step=state.global_step,
                )

            self.model.train()
            print("=" * 50 + "\n")
