# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# setup.py is the fallback installation script when pyproject.toml does not work
import os
from pathlib import Path

from setuptools import find_packages, setup

# Try to read version from agent_system/version if it exists, otherwise default
version_folder = os.path.dirname(os.path.join(os.path.abspath(__file__)))
version_file = os.path.join(version_folder, "agent_system", "version", "version")
if os.path.exists(version_file):
    with open(version_file) as f:
        __version__ = f.read().strip()
else:
    __version__ = "0.1.0"

install_requires = [
    "accelerate",
    "codetiming",
    "datasets",
    "dill",
    "hnswlib",
    "hydra-core",
    "numpy",
    "openai",
    "pandas",
    "peft",
    "pyarrow>=19.0.0",
    "pybind11",
    "pylatexenc",
    "ray[default]>=2.41.0,<=2.50.0",
    "scikit-learn",
    "torchdata",
    "tqdm",
    "tensordict>=0.8.0,<=0.10.0,!=0.9.0",
    "transformers<=4.57.3",
    "wandb",
    "packaging>=20.0",
    "qwen-vl-utils[decord]",
]

TEST_REQUIRES = ["pytest", "pre-commit", "py-spy"]
GPU_REQUIRES = ["liger-kernel", "flash-attn"]
VLLM_REQUIRES = ["tensordict>=0.8.0,<=0.10.0,!=0.9.0", "vllm>=0.8.5,<=0.11.0"]
SGLANG_REQUIRES = [
    "tensordict>=0.8.0,<=0.10.0,!=0.9.0",
    "sglang[srt,openai]==0.5.5",
    "torch==2.8.0",
]
TRL_REQUIRES = ["trl<=0.9.6"]
SERVING_REQUIRES = ["fastapi", "uvicorn", "httpx"]

extras_require = {
    "test": TEST_REQUIRES,
    "gpu": GPU_REQUIRES,
    "vllm": VLLM_REQUIRES,
    "sglang": SGLANG_REQUIRES,
    "trl": TRL_REQUIRES,
    "serving": SERVING_REQUIRES,
}


this_directory = Path(__file__).parent
readme_path = this_directory / "README.md"
long_description = readme_path.read_text() if readme_path.exists() else "CAEI"

setup(
    name="caei-agent",
    version=__version__,
    package_dir={"": "."},
    packages=find_packages(where=".", exclude=["tests", "tests.*", "examples", "examples.*"]),
    license="Apache 2.0",
    author="CAEI Contributors",
    description="CAEI：Cognitive-Aware Experience Internalization for Self-Evolving Agents",
    install_requires=install_requires,
    extras_require=extras_require,
    include_package_data=True,
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.9",
)
