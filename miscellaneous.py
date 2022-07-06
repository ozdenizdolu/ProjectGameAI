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
    epsilon = torch.ones_like(output)*(10**-20)
    
    return (-1.) * torch.mean(
        torch.sum(torch.log(output + epsilon) * target, dim = 1),
        dim = 0)
