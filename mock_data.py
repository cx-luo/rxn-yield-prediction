# -*- coding: utf-8 -*-
# @Time    : 2025/9/24 17:26
# @Author  : chengxiang.luo
# @Email   : chengxiang.luo@foxmail.com
# @File    : mock_data.py
# @Software: PyCharm
import random


def generate_reaction(reaction_id):
    """生成一条随机反应数据"""
    return {
        'reaction_id': reaction_id,
        'eqv1': round(random.uniform(0.1, 2.0), 4),  # 0.1 ~ 2.0
        'eqv2': round(random.uniform(0.1, 2.0), 4),
        'eqv3': round(random.uniform(0.0, 1.0), 4),  # 可选
        'eqv4': round(random.uniform(0.0, 1.0), 4),  # 可选
        'eqv5': round(random.uniform(0.0, 1.0), 4),  # 可选
        'reaction_temperature': round(random.uniform(20, 150), 2),  # 20 ~ 150°C
        'time': int(random.uniform(10, 300)),  # 10 ~ 300 分钟
        'product1_yield': round(random.uniform(50, 99), 3),  # 50% ~ 99%
        'product2_yield': round(random.uniform(30, 99), 3)  # 30% ~ 99%
    }
