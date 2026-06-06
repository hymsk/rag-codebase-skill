# RAG 代码解析器包
# 按语言组织，当前支持: cpp / python / go
#
# 扩展新语言:
#   1. 在此目录下创建 <language>.py
#   2. 实现 parse_file(path) -> List[Dict] 接口
#   3. 实现 collect_files(root) -> List[str] 接口
#   4. 实现 should_skip_file(path) -> bool 接口
