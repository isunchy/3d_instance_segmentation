import os
import sys
import numpy as np


category_info = {
  'Chair': {
    '#part': {
      1: 6,
      2: 30,
      3: 39
    }
  },
  'Table': {
    '#part': {
      1: 11,
      2: 42,
      3: 51
    }
  },
  'Lamp': {
    '#part': {
      1: 18,
      2: 28,
      3: 41
    }
  },
  'StorageFurniture': {
    '#part': {
      1: 7,
      2: 19,
      3: 24
    }
  },
  'Bed': {
    '#part': {
      1: 4,
      2: 10,
      3: 15
    }
  },
  'Bag': {
    '#part': {
      1: 4
    }
  },
  'Bottle': {
    '#part': {
      1: 6,
      3: 9
    }
  },
  'Bowl': {
    '#part': {
      1: 4
    }
  },
  'Clock': {
    '#part': {
      1: 6,
      3: 11
    }
  },
  'Dishwasher': {
    '#part': {
      1: 3,
      2: 5,
      3: 7
    }
  },
  'Display': {
    '#part': {
      1: 3,
      3: 4
    }
  },
  'Door': {
    '#part': {
      1: 3,
      2: 4,
      3: 5
    }
  },
  'Earphone': {
    '#part': {
      1: 6,
      3: 10
    }
  },
  'Faucet': {
    '#part': {
      1: 8,
      3: 12
    }
  },
  'Hat': {
    '#part': {
      1: 6
    }
  },
  'Keyboard': {
    '#part': {
      1: 3
    }
  },
  'Knife': {
    '#part': {
      1: 5,
      3: 10
    }
  },
  'Laptop': {
    '#part': {
      1: 3
    }
  },
  'Microwave': {
    '#part': {
      1: 3,
      2: 5,
      3: 6
    }
  },
  'Mug': {
    '#part': {
      1: 4
    }
  },
  'Refrigerator': {
    '#part': {
      1: 3,
      2: 6,
      3: 7
    }
  },
  'Scissors': {
    '#part': {
      1: 3
    }
  },
  'TrashCan': {
    '#part': {
      1: 5,
      3: 11
    }
  },
  'Vase': {
    '#part': {
      1: 4,
      3: 6
    }
  }
}


def load_metric(filename):
  assert os.path.isfile(filename), filename
  content = open(filename).readlines()
  prefix_list = ['mAP50']
  # prefix_list = ['miou v2']
  # prefix_list = ['mAP25']
  # prefix_list = ['mAP75']
  # prefix_list = ['shape mAP']
  # prefix_list = ['miou v3']
  metric = []
  for prefix in prefix_list:
    for line in content:
      if line.startswith(' {}'.format(prefix)):
        metric.append(float(line.split()[-1]))
  return metric




def main():
  folder = os.path.join('log', 'PartNet')
  assert os.path.isdir(folder), folder
  category_list = os.listdir(folder)
  n_cat = len(category_list)
  metric_list = -np.ones([3, n_cat], np.float32)
  print('n_cat: {}'.format(n_cat))
  for category in category_list:
    category_folder = os.path.join(folder, category)
    filename = [file for file in os.listdir(category_folder) if file.endswith('.txt')][0]
    metric = load_metric(os.path.join(category_folder, filename))

    for level in [1, 2, 3]:
      if level in category_info[category]['#part']:
        metric_list[level-1][category_list.index(category)] = metric[level-1]

  for category in category_list:
    print('{:>5} '.format(category[:5]), end='')
  print(''); sys.stdout.flush()
  for level in [1, 2, 3]:
    for i in range(n_cat):
      value = metric_list[level-1][i]
      if value >= 0:
        print('{:5.1f},'.format(value), end='')
      else:
        print('     ,', end='')
    print('')


if __name__ == '__main__':
  main()