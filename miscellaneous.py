import math
import random
import torch

def after(f,g):
    return lambda x: f(g(x))

class PDist:
    
    def __init__(self, items, values, normalize = False, supports_hash = False):
        items = list(items)
        values = list(values)
        if len(items) != len(values):
            raise ValueError('Number of items and probabilities do not match')       
        if len(values) == 0:
            raise ValueError('No probabilities are given')
        if any(value < 0 for value in values):
            raise ValueError('Some probabilities are negative')
        if not normalize:
            if not math.isclose(sum(values), 1.):
                raise ValueError('Sum of Probabilities is not 1.')
        else:
            sum_ = sum(values)
            values = [value/sum_ for value in values]
        
        self._items = items
        self._values = values
        self._supports_hash = supports_hash
        if supports_hash:
            self._dict = {item: value for item,value in zip(items,values)}
        
    def sample(self):
        return random.choices(self._items, self._values, k = 1)[0]
    
    def __getitem__(self, item):
        if not self._supports_hash:
            raise AttributeError('''This PDist's items cannot be hashed.''')
        return self._dict[item]
    
    def __iter__(self):
        return iter(self._items)
    
    def __len__(self):
        return len(self._items)
    
    def __contains__(self, item):
        if self._supports_hash:
            return item in self._dict
        else:
            return item in self._items
    
    def __str__(self):
        return str([(item, probability) for item,probability in zip(
                                            self._items,self._values)])
    def __repr__(self):
        return str(self)
    
    def items(self):
        return zip(self._items, self._values)

  
def cross_entropy(output, target):
    """
    Return the cross entropy of output distribution relative to target
    distribution.
    
    Output and target is assumed to be of shape (N, M) where 
    the first dimension is the index of each distinct problem. The function
    computes the cross entropy for each N probability distributions and
    returns the mean
    
    Parameters
    ----------
    output : tensor
        The probability distribution used to approximate the target.
    target : tensor
        The ground truth.

    Returns
    -------
    The cross entropy in the form of tensor.

    """
    # epsilon = torch.ones_like(output)*(10**-20)
    
    if not (len(output.shape) == len(target.shape) == 2):
        raise ValueError('Invalid input to cross_entropy.')
    if not output.shape == target.shape:
        raise ValueError('Dimensions do not match.')
    
    return torch.mul(-1., torch.mean(
        torch.sum(torch.mul(torch.log(torch.add(output, 10**-20)), target), dim = 1),
        dim = 0))

def generate_init_code(*attributes):
    """A Macro for coding. """
    
    return ''.join(['self._{0} = {0}\n'.format(attribute)
                    for attribute in attributes])

