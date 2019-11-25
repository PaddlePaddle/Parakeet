from parakeet.datasets import vctk
from pathlib import Path
from parakeet.data.datacargo import DataCargo

root = Path("/workspace/datasets/VCTK-Corpus")
vctk_dataset = vctk.VCTK(root)
vctk_cargo = DataCargo(vctk_dataset, batch_size=16, shuffle=True, drop_last=True)

for i, batch in enumerate(vctk_cargo):
    print(i)

