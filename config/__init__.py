import os
import importlib


def register_upscalers():
    upscalers_dir = os.path.join(os.path.dirname(__file__), '..', 'upscalers')

    for subdir in os.listdir(upscalers_dir):
        subdir_path = os.path.join(upscalers_dir, subdir)
        if os.path.isfile(subdir_path):
            continue

        files = os.listdir(subdir_path)

        for file in files:
            if file.endswith(".py") and file != "__init__.py":
                module_name = file.replace('.py', '')
                import_name = f'upscalers.{subdir}.{module_name}'
                importlib.import_module(import_name)


register_upscalers()

