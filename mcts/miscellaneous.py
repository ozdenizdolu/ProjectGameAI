# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 01:04:07 2022

@author: Fırat
"""


def after(f,g):
    return lambda x: f(g(x))