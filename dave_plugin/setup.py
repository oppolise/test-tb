from setuptools import setup, find_packages

setup(
    name="tensorboard-plugin-dave",
    version="1.0.0",
    description="A minimalist profiler plugin to display operator trees.",
    packages=find_packages(),
    package_data={
        "dave_plugin": ["static/**"],
    },
    entry_points={
        "tensorboard_plugins": [
            "dave = dave_plugin.plugin:DavePlugin",
        ],
    },
)
