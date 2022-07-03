import math
import random

def after(f,g):
    return lambda x: f(g(x))

class PDist:
    
    def __init__(self, items, values, normalize = False, supports_hash = False):
        items = list(items)
        values = list(values)
        if len(items) != len(values):
            raise ValueError('Number of items and probabilities do not match')       
        if len(values) == 0:
            raise ValueError('No probabilities is given')
        if any(value < 0 for value in values):
            raise ValueError('Some probabilities are negative')
        if not normalize:
            if not math.isclose(sum(values), 1.):
                raise ValueError('Sum of Probabilities is not 1.')
        else:
            sum_ = sum(values)
            values = [value/sum_ for value in values]
        
        self.items = items
        self.values = values
        self._supports_hash = supports_hash
        if supports_hash:
            self._dict = {item: value for item,value in zip(items,values)}
        
    def sample(self):
        return random.choices(self.items, self.values, k = 1)[0]
    
    def __getitem__(self, item):
        if not self._supports_hash:
            raise AttributeError('''PDist's items cannot be hashed.''')
        return self._dict[item]
    
    def __iter__(self):
        return iter(self.items)
    
    def __len__(self):
        return len(self.items)
    
    def __contains__(self, item):
        if self._supports_hash:
            return item in self._dict
        else:
            return item in self.items