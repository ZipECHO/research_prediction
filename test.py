# # coding=utf-8
# # !/usr/bin/python
#
# # 接口请求示例为：http://open.api.tianyancha.com/services/open/jr/bankruptcy/detail/2.0?gid=365223910&uuid=4905eef8fc704bcfbb82b6f0294d6fc5
#
# # pip install requests
# import requests
# import time
# import hashlib
# import json
#
# #  token可以从 数据中心 -> 我的接口 中获取
# token = "10e41df4-d30c-4d14-a31a-ab708ddb6f9b"
# encode = 'utf-8'
#
# url = "http://open.api.tianyancha.com/services/open/jr/bankruptcy/detail/2.0?gid=365223910&uuid=4905eef8fc704bcfbb82b6f0294d6fc5"
# headers = {'Authorization': token}
# response = requests.get(url, headers=headers)
#
# # 结果打印
# print(response.status_code)
# print(response.text)

# coding=utf-8
# !/usr/bin/python

# 接口请求示例为：http://open.api.tianyancha.com/services/open/jr/bankruptcy/2.0?pageSize=20&keyword=长沙新世界国际大饭店有限公司&pageNum=1

# pip install requests
import requests
import time
import hashlib
import json

#  token可以从 数据中心 -> 我的接口 中获取
token = "您的token"
encode = 'utf-8'

url = "http://open.api.tianyancha.com/services/open/jr/bankruptcy/2.0?pageSize=20&keyword=长沙新世界国际大饭店有限公司&pageNum=1"
headers = {'Authorization': token}
response = requests.get(url, headers=headers)

# 结果打印
print(response.status_code)
print(response.text)