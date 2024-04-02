from typing import Union, Callable, Any, Sequence
import copy
import torch.nn as nn


class RepNet(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        condition_func: Callable[[nn.Module, str], bool],
        replacement_func: Callable[[nn.Module], nn.Module],
        *args: Any,
        **kwargs: Any,
    ):
        """Abstract class for replacing modules in models.

        Args:
            model: base model to be modified
            condition_func: function that takes a module and returns a boolean (True if module should be replaced, False otherwise)
            replacement_func: function that takes a module and returns a new module

        """
        super().__init__(*args, **kwargs)
        # Set _base_model as one of the first operations
        self._base_model = copy.deepcopy(model)
        self.base_model = copy.deepcopy(model)
        self.condition_func = condition_func
        self.replacement_func = replacement_func
        self.replacements = {}
        assert hasattr(self, "_base_model"), "_base_model has not been set in RepNet."

    def replace_base_model_layers(self):
        self._replace__base_model_layers(self.condition_func, self.replacement_func)

    def __getattr__(self, name: str) -> Any:
        # Try to get from self first (e.g., attributes added in __init__)
        try:
            return super().__getattr__(name)
        except AttributeError:
            # Try to get from _base_model if not found in self
            return getattr(self._base_model, name)

    def replacement_report(self):
        # Print replacement report
        print("Replacement report:")
        print(f"{'Layer':<30} {'Original':<30} {'Replacement':<30}")
        for addr, replacement in self.replacements.items():
            print(
                f"{addr:<30} {type(self._get_module(self._base_model, addr)):<30} {type(replacement):<30}"
            )

    def _replace__base_model_layers(
        self,
        condition_func: Callable[[nn.Module, str], bool],
        replacement_func: Callable[[nn.Module], nn.Module],
    ) -> "RepNet":
        """Replace modules that satisfy a condition using a replacement function

        Args:
            condition_func: function that takes a module and returns a boolean (True if module should be replaced, False otherwise)
            replacement_func: function that takes a module and returns a new module

        Returns:
            self

        """
        for name, layer in self._base_model.named_modules():
            if condition_func(layer, name):
                replacement_module = replacement_func(layer)
                replacement_address = self._parse_model_addr(name)
                self.replacements[name] = replacement_module
                self._set_module(
                    self._base_model, replacement_address, replacement_module
                )
        return self

    @staticmethod
    def get_num_params(model: nn.Module) -> dict:
        params_dict = {k: 0 for k in ["trainable", "fixed", "total"]}
        for p in model.parameters():
            params_dict["total"] += p.numel()
            if p.requires_grad:
                params_dict["trainable"] += p.numel()
            else:
                params_dict["fixed"] += p.numel()

        params_dict = {k: v / 1e6 for k, v in params_dict.items()}
        return params_dict

    def _get_module(
        self,
        parent: Union[nn.Module, nn.Sequential, nn.ModuleList],
        replacement_addr_list: list,
    ) -> nn.Module:
        """Recursive function used to access child modules from a parent nn.Module object

        Args:
            replacement_addr_list: specifies how to access target object from ancestor object.
                ['layer1', 0, 'conv2']

        Returns:
            target object/layer to be replaced.

        """

        if len(replacement_addr_list) == 0:
            return parent
        else:
            attr = replacement_addr_list.pop(0)
            if isinstance(attr, int):
                # Explicitly handle indexing for nn.Sequential or nn.ModuleList
                if isinstance(parent, (nn.Sequential, nn.ModuleList)):
                    child = parent[attr]
                else:
                    raise TypeError(
                        f"Index access attempted on non-indexable module: {type(parent)}"
                    )
            else:
                # Use getattr for named children access
                child = getattr(parent, attr)
            return self._get_module(child, replacement_addr_list)

    def _set_module(
        self,
        model: nn.Module,
        replacement_addr_list: list[Union[int, str]],
        replacement_layer: nn.Module,
    ) -> None:
        """Sets attribute of `model` accessed via `replacement_addr_list` to `replacement_layer`."""
        # Navigate to the parent module of the target.
        parent = self._get_module(model, replacement_addr_list[:-1])
        target = replacement_addr_list[-1]

        if isinstance(target, int):
            # Handle indexed children for nn.Sequential and nn.ModuleList
            if isinstance(parent, (nn.Sequential, nn.ModuleList)):
                parent[target] = replacement_layer
            else:
                raise TypeError(
                    f"Parent module does not support indexed assignment: {type(parent)}"
                )
        else:
            # Use setattr for named children access
            setattr(parent, target, replacement_layer)

    @staticmethod
    def _parse_model_addr(access_str: str) -> list[Union[int, str]]:
        """Parses path to child from a parent. E.g., layer1.0.conv2 ==> ['layer1', 0, 'conv2']"""
        parsed: Sequence[Union[int, str]] = access_str.split(".")
        for i in range(len(parsed)):
            try:
                parsed[i] = int(parsed[i])  # type: ignore
            except ValueError:
                pass  # Remain as string if conversion fails
        return parsed  # type: ignore

    def forward(self, *args, **kwargs):
        return self._base_model(*args, **kwargs)
