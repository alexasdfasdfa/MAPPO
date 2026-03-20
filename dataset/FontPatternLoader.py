import json
import os
import random
from typing import List, Dict, Union, Tuple, Optional, Any

class FontPatternLoader:
    def __init__(self, target_lengths: Optional[Union[int, List[int]]] = None, dataset_dir: str = "./dataset"):
        """
        初始化加载器。
        
        :param target_lengths: 需要加载的长度列表。
                               - int: 只加载该长度 (例如 10 -> 读取 10.json)
                               - list: 加载列表中的长度 (例如 [10, 20])
                               - None: 自动加载 dataset_dir 下所有符合命名规则的数字.json 文件
        :param dataset_dir: 数据集目录，默认为 "./dataset"
        """
        self.data_store: Dict[int, List[Dict[str, Any]]] = {}
        self.dataset_dir = dataset_dir
        
        if not os.path.exists(self.dataset_dir):
            print(f"警告：数据集目录 '{self.dataset_dir}' 不存在。")
            return

        # 1. 确定需要加载哪些长度
        lengths_to_load = []
        
        if target_lengths is None:
            # 自动扫描目录下所有 {数字}.json 文件
            for filename in os.listdir(self.dataset_dir):
                if filename.endswith('.json'):
                    name_part = filename[:-5] # 去掉 .json
                    if name_part.isdigit():
                        lengths_to_load.append(int(name_part))
        elif isinstance(target_lengths, int):
            lengths_to_load = [target_lengths]
        elif isinstance(target_lengths, list):
            lengths_to_load = target_lengths
        else:
            raise ValueError("target_lengths 必须是 int, list 或 None")

        # 2. 执行加载
        for length in lengths_to_load:
            file_path = os.path.join(self.dataset_dir, f"{length}.json")
            
            if not os.path.exists(file_path):
                print(f"提示：未找到文件 {file_path}，跳过长度 {length}。")
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = json.load(f)
                
                # 验证数据格式：必须是列表
                if not isinstance(content, list):
                    print(f"错误：{file_path} 的内容不是列表，已跳过。")
                    continue
                
                # 存入内存
                # 确保每个元素都是字典，且包含必要字段 (可选验证)
                valid_items = []
                for item in content:
                    if isinstance(item, dict):
                        # 这里假设数据结构为 {'name':..., 'coordinates':...}
                        # 即使没有 'len' 字段也没关系，因为长度由文件名决定
                        valid_items.append(item)
                    else:
                        print(f"警告：在 {file_path} 中发现非字典项，已忽略。")
                
                if valid_items:
                    self.data_store[length] = valid_items
                    print(f"成功加载长度 {length}: {len(valid_items)} 个图案。")
                else:
                    print(f"警告：{file_path} 中没有有效数据。")

            except json.JSONDecodeError as e:
                print(f"JSON 解析失败 {file_path}: {e}")
            except Exception as e:
                print(f"读取文件 {file_path} 时发生未知错误: {e}")

        if not self.data_store:
            print("初始化完成：未加载到任何数据。")
        else:
            print(f"初始化完成：内存中可用长度 -> {sorted(self.data_store.keys())}")

    def shuffle(self, length: int):
        """
        随机重新排序指定长度下的所有图案 (In-place shuffle)。
        
        :param length: 要重排的图案长度。
        """
        if length not in self.data_store:
            print(f"错误：内存中没有长度为 {length} 的数据，无法重排。")
            return
        
        patterns = self.data_store[length]
        if len(patterns) <= 1:
            return
            
        random.shuffle(patterns)
        # print(f"长度为 {length} 的 {len(patterns)} 个图案已随机重排。")

    def get(self, length: int, param2: Union[int, Tuple[int, int]]) -> Union[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        获取图案数据。
        
        :param length: 图案长度 (对应文件名)。
        :param param2: 
            - int: 返回该索引处的单个图案字典 {'name':..., 'coordinates':...}
            - tuple (start, end): 返回切片范围内的图案列表
        :return: 单个字典 或 字典列表。如果未找到返回 None 或空列表。
        """
        if length not in self.data_store:
            return None if isinstance(param2, int) else []
        
        patterns = self.data_store[length]
        
        if isinstance(param2, int):
            try:
                return patterns[param2]
            except IndexError:
                return None
                
        elif isinstance(param2, tuple) and len(param2) == 2:
            start, end = param2
            return patterns[start:end]
        
        else:
            raise ValueError("param2 必须是 int (索引) 或 tuple (start, end)")

    def __getitem__(self, key):
        """
        支持方括号访问的核心方法。
        
        用法示例:
        1. loader[10]          -> 返回长度为 10 的整个列表
        2. loader[10, 0]       -> 返回长度为 10 的第 0 个图案 (字典)
        3. loader[10, (0, 5)]  -> 返回长度为 10 的前 5 个图案 (列表切片)
        4. loader[10, 1:5]     -> (高级) 如果传入 slice 对象，也支持切片
        """
        # 情况 A: 单键访问 -> loader[10]
        if isinstance(key, int):
            if key not in self.data_store:
                raise KeyError(f"长度 {key} 不存在于数据集中。")
            return self.data_store[key]
        
        # 情况 B: 复合键访问 -> loader[10, 0] 或 loader[10, (0, 5)]
        elif isinstance(key, tuple):
            if len(key) != 2:
                raise ValueError("方括号内元组必须包含两个元素: (length, index_or_slice)")
            
            length, selector = key
            
            if length not in self.data_store:
                raise KeyError(f"长度 {length} 不存在于数据集中。")
            
            patterns = self.data_store[length]
            
            # 子情况 B1: 选择器是整数 (索引) -> loader[10, 0]
            if isinstance(selector, int):
                try:
                    return patterns[selector]
                except IndexError:
                    raise IndexError(f"长度 {length} 的数据集中没有索引 {selector}。")
            
            # 子情况 B2: 选择器是元组 (范围) -> loader[10, (0, 5)]
            elif isinstance(selector, tuple) and len(selector) == 2:
                start, end = selector
                return patterns[start:end]
            
            # 子情况 B3: 选择器是 slice 对象 (支持语法 loader[10, 0:5]) -> 虽然用户通常传元组，但支持 slice 更 Pythonic
            elif isinstance(selector, slice):
                return patterns[selector]
            
            else:
                raise TypeError(f"不支持的选择器类型: {type(selector)}。请使用 int 或 (start, end) 元组。")
        
        else:
            raise TypeError(f"无效的键类型: {type(key)}。请使用 int 或 (int, int/tuple) 。")

    def __len__(self):
        """返回已加载的不同长度的数量。"""
        return len(self.data_store)

    def __repr__(self):
        """打印对象的简要信息。"""
        lengths = sorted(self.data_store.keys())
        counts = {k: len(v) for k, v in self.data_store.items()}
        return f"<FontPatternLoader loaded lengths: {lengths}, counts: {counts}>"
