def _init():  # 初始化
    global glo_dict
    glo_dict = {}

def set_value(key, value):
    glo_dict[key] = value

def get_value(key):
    if key in glo_dict:
        return glo_dict[key]
    else:
        return None

def printvar():
    for kv in glo_dict.items():
        print(kv)


