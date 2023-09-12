import numpy as np
import collections.abc
from Model.layers.linear import LinearLayer
from Model.layers.bias import BiasLayer



def is_modules_with_parameters(value):
    print('that here', value)
    return isinstance(value, LinearLayer) or isinstance(value, BiasLayer)


class ModuleList(collections.abc.MutableSequence):
    def __init__(self, *args):
        self.list = list()
        self.list.extend(list(args))
        # print('List of MODS', self.list)
        pass

    def __getitem__(self, i):
        return self.list[i]

    def __setitem__(self, i, v):
        self.list[i] = v

    def __delitem__(self, i):
        del self.list[i]
        pass

    def __len__(self):
        return len(self.list)

    def insert(self, i, v):
        self.list.insert(i, v)
        pass

 
    def get_modules_with_parameters(self):
        modules_with_parameters_list = []
        for mod in self.modules_with_parameters:
            print(f"Checking module: {mod}")
            if is_modules_with_parameters(mod):
                print(f"Adding module: {mod}")
                modules_with_parameters_list.append(mod)
        print(f"Final list of modules with parameters: {modules_with_parameters_list}")
        return modules_with_parameters_list

    pass


class BaseNetwork:
    def __init__(self):
        super().__setattr__("initialized", True)
        super().__setattr__("modules_with_parameters", [])
        super().__setattr__("output_layer", None)

    def set_output_layer(self, layer):
        super().__setattr__("output_layer", layer)
        pass

    def get_output_layer(self):
        return self.output_layer

    def __setattr__(self, name, value):
        print(f"__setattr__ called with name: {name} and value: {value} type: {type(value)}")

        if not hasattr(self, "initialized") or (not self.initialized):
            print("Initialization condition failed.")
            raise RuntimeError("You must call super().__init__() before assigning any layer in __init__().")
        print("Initialization condition passed.")
        if is_modules_with_parameters(value) or isinstance(value, ModuleList):
            print("Module with parameters identified.")
            self.modules_with_parameters.append(value)
        super().__setattr__(name, value)


    def get_modules_with_parameters(self):
        modules_with_parameters_list = []
        for mod in self.modules_with_parameters:
            if isinstance(mod, ModuleList):
                modules_with_parameters_list.extend(mod.get_modules_with_parameters())
                pass
            else:
                modules_with_parameters_list.append(mod)
                pass
            pass
        return modules_with_parameters_list

    def forward(self):
        return self.output_layer.forward()

    def backward(self, input_grad):
        self.output_layer.backward(input_grad)
        pass

    def state_dict(self):
        all_params = []
        for m in self.get_modules_with_parameters():
            all_params.append(m.W)
            pass
        return all_params

    def load_state_dict(self, state_dict):
        assert len(state_dict) == len(self.get_modules_with_parameters())
        for m, lw in zip(self.get_modules_with_parameters(), state_dict):
            m.W = lw
            pass
        pass

    pass
