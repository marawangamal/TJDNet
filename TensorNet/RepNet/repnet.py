from typing import Callable, Any, Sequence
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
        self.base_model = model
        self._replace_base_model_layers(condition_func, replacement_func)

    def _replace_base_model_layers(
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
        for name, layer in self.base_model.named_modules():
            if condition_func(layer, name):
                replacement_module = replacement_func(layer)
                replacement_address = self._parse_model_addr(name)
                self._set_module(
                    self.base_model, replacement_address, replacement_module
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
        parent: nn.Module | nn.Sequential | nn.ModuleList,
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

    # def _set_module(
    #     self,
    #     model: nn.Module | nn.Sequential | nn.ModuleList,
    #     replacement_addr_list: list[int | str],
    #     replacement_layer: nn.Module,
    # ) -> None:
    #     """Sets attribute of `model` accessed via `replacement_addr_list` to `replacement_layer`"""
    #     if isinstance(replacement_addr_list[-1], int):

    #         if isinstance(model, (nn.Sequential, nn.ModuleList)):
    #             self._get_module(model, replacement_addr_list[:-1])[
    #                 replacement_addr_list[-1]
    #             ] = replacement_layer
    #         else:
    #             raise TypeError(
    #                 f"Index access attempted on non-indexable module: {type(parent)}"
    #             )
    #     else:
    #         setattr(
    #             self._get_module(model, replacement_addr_list[:-1]),
    #             replacement_addr_list[-1],
    #             replacement_layer,
    #         )

    def _set_module(
        self,
        model: nn.Module,
        replacement_addr_list: list[int | str],
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
    def _parse_model_addr(access_str: str) -> list[int | str]:
        """Parses path to child from a parent. E.g., layer1.0.conv2 ==> ['layer1', 0, 'conv2']"""
        parsed: Sequence[int | str] = access_str.split(".")
        for i in range(len(parsed)):
            try:
                parsed[i] = int(parsed[i])  # type: ignore
            except ValueError:
                pass  # Remain as string if conversion fails
        return parsed  # type: ignore

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)
