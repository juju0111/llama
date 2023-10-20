import os
import re
import torch
from tqdm.cli import tqdm

path_70b = './llama-2-70b-chat/'

print(sorted(os.listdir(path_70b)))

weights = {
  int(fn.split('.')[1]): torch.load(f'{path_70b}{fn}', map_location=torch.device('cpu'))
  for fn in tqdm(sorted(os.listdir(path_70b)))
  if fn.endswith('.pth')
}


not_distributed = {
  k 
  for k in weights[0].keys()
  if all((weights[0][k] == weights[i][k]).min() for i in range(1,8))
}

merge_dimensions ={
  r'^layers.\d+.attention.wq.weight$': 0,
  r'^layers.\d+.attention.wk.weight$': 0,
  r'^layers.\d+.attention.wv.weight$': 0,
  r'^layers.\d+.attention.wo.weight$': 1,

  r'^tok_embeddings.weight$': 1,

  r'^layers.\d+.feed_forward.w1.weight$': 0,
  r'^layers.\d+.feed_forward.w2.weight$': 1,
  r'^layers.\d+.feed_forward.w3.weight$': 0,
  r'^output.weight$': 0 
}

# Which files are merged into one
merge_groups = [[0,1,2],[3,4,5],[6,7]]

# Merging (or copying if not distributed)
output_weights = {}
for output, group in enumerate(merge_groups):
  output_weights[output] = dict()
  for name in tqdm(weights[group[0]], leave=False):
    if name in not_distributed:
      output_weights[output][name] = weights[0][name]
    else:
      axis = next(axis for exp, axis in merge_dimensions.items() if re.match(exp, name))
      output_weights[output][name] = torch.cat([
          weights[member][name]
          for member in group
      ], axis=axis)

path_70b = './llama-2-70b-chat'

os.makedirs(f'{path_70b}/three-nodes/', exist_ok=True)
with open(f'{path_70b}/params.json') as fin:
  with open(f'{path_70b}/three-nodes/params.json', 'w') as fout:
    fout.write(fin.read())


torch.save(
    output_weights[0],
    f'{path_70b}/three-nodes/consolidated.00.pth'
)
torch.save(
    output_weights[1],
    f'{path_70b}/three-nodes/consolidated.01.pth'
)

torch.save(
    output_weights[2],
    f'{path_70b}/three-nodes/consolidated.02.pth'
)
# torch.save(
#     output_weights[3],
#     f'{path_70b}/three-nodes/consolidated.03.pth'
# )