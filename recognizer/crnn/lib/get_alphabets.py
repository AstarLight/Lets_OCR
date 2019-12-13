#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = 'changxin'
__mtime__ = '2019/3/14'
"""


def get_alphabets(filelists):
    a = []
    for x in filelists:
        with open(x) as f:
            strings = f.readlines()
            string = [y.strip().split(' ')[1] for y in strings]
            tmp = []
            for z in ''.join(string):
                tmp.append(z)
            a = a + tmp
    with open('alphabets.py', 'w') as e:
        e.write("#coding=utf8\n")
        e.write('\"\"\"' + ''.join(list(set(a))) + '\"\"\"')
