import yaml
import os
from collections import Counter

if __name__ == '__main__':
    c = Counter()
    total = 0
    with open('nlu.yml', 'r') as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    i = 0
    for intent_data in data['nlu']:
        try:
            dest = os.path.join('data', intent_data['intent'])
            os.makedirs(dest, exist_ok=True)
            for n, example in enumerate(intent_data['examples'].split('\n')):
                c.update([intent_data['intent']])
                total += 1
                if len(example.strip('-').strip()) > 0:
                    with open(os.path.join(dest, f'{n}.txt'), 'wt') as f:
                        print(example.strip('- '), file=f)
            i += 1
        except:
            print(intent_data)
    print(c.most_common(5), total, c.most_common(1)[0][1] / total)
