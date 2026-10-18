



import torch

# 你的 key_fn 工厂函数
def _get_key_fn(page_size: int):
    if page_size == 1:
        return lambda x: x[0].item()
    return lambda x: tuple(x[:page_size].tolist())


# ========== 测试 1: page_size = 1 ==========
print("===== page_size=1 =====")
key_fn_1 = _get_key_fn(page_size=1)
tokens_1 = torch.tensor([101, 102, 103, 104])
k1 = key_fn_1(tokens_1)
print(f"input tensor: {tokens_1}")
print(f"key: {k1}, type={type(k1)}\n")

# ========== 测试 2: page_size = 4 ==========
print("===== page_size=4 =====")
key_fn_4 = _get_key_fn(page_size=4)
tokens_4 = torch.tensor([101, 102, 103, 104, 105, 106])
k4 = key_fn_4(tokens_4)
print(f"input tensor: {tokens_4}")
print(f"key: {k4}, type={type(k4)}\n")

# ========== 演示放到 dict.children 里，模拟 radix tree 查询 ==========
print("===== 模拟 node.children dict 查询 =====")
# 构造子节点字典，key就是用同样key_fn生成
children = {
    (101,102,103,104): "RadixNode A",
    (201,202,203,204): "RadixNode B",
}

# 输入待匹配token，计算key去get
input_ids = torch.tensor([101,102,103,104, 999,999])
search_key = key_fn_4(input_ids)
found_node = children.get(search_key)
print(f"search_key = {search_key}")
print(f"get result: {found_node}")

# 换一个不匹配的token
input_ids_bad = torch.tensor([666,102,103,104])
search_key_bad = key_fn_4(input_ids_bad)
found_node_bad = children.get(search_key_bad)
print(f"bad search_key = {search_key_bad}")
print(f"get result: {found_node_bad}")
