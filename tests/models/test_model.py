import unittest
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer


class TestModels(unittest.TestCase):
    def test_model_pad_llama(self):
        model_name = "meta-llama/Llama-2-7b-chat-hf"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        input_ids = torch.randint(0, tokenizer.vocab_size, (1, 15), device=device)

        kw = {"device": device, "dtype": input_ids.dtype}

        x = torch.randint(0, 100, (1, 15), **kw)
        x_pad = torch.cat((torch.zeros(1, 5, **kw), x), dim=1)
        x_pad_mask = torch.cat(
            (torch.zeros(1, 5, **kw), torch.ones(1, 15, **kw)), dim=1
        )

        y = model.model(input_ids=x).last_hidden_state
        y_pad = model.model(
            input_ids=x_pad, attention_mask=x_pad_mask
        ).last_hidden_state
        torch.allclose(y[0, 0], y_pad[0, 5], atol=1e-4)
        torch.allclose(y[0, -1], y_pad[0, -1], atol=1e-4)

    def test_model_pad_gpt2(self):
        model_name = "gpt2"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        input_ids = torch.randint(0, tokenizer.vocab_size, (1, 15), device=device)

        kw = {"device": device, "dtype": input_ids.dtype}

        x = torch.randint(0, 100, (1, 15), **kw)
        x_pad = torch.cat((torch.zeros(1, 5, **kw), x), dim=1)
        x_pad_mask = torch.cat(
            (torch.zeros(1, 5, **kw), torch.ones(1, 15, **kw)), dim=1
        )

        y = model.transformer(input_ids=x).last_hidden_state
        y_pad = model.transformer(
            input_ids=x_pad, attention_mask=x_pad_mask
        ).last_hidden_state
        # position_ids = (
        #     torch.cat((torch.arange(15, 20), torch.arange(0, 15)))
        #     .reshape(1, -1)
        #     .to(device)
        # )
        # expected
        torch.allclose(y[0, 0], y_pad[0, 5], atol=1e-4)
        torch.allclose(y[0, -1], y_pad[0, -1], atol=1e-4)


if __name__ == "__main__":
    unittest.main()


# position_ids = torch.cat((torch.arange(15,20), torch.arange(0, 15))).reshape(1, -1)
