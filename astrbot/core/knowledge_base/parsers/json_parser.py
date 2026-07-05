"""JSON 文件解析器

将 JSON 文件内容格式化为可读的文本块。
"""

import json

from astrbot.core.knowledge_base.parsers.base import BaseParser, ParseResult


class JsonParser(BaseParser):
    """JSON 解析器

    将 JSON 文件解析为规范化、带缩进的文本，便于作为文本块进行索引。
    """

    async def parse(self, file_content: bytes, file_name: str) -> ParseResult:
        """解析 JSON 文件。

        Args:
            file_content: 文件二进制内容
            file_name: 文件名，仅用于报错信息

        Returns:
            ParseResult: 包含格式化 JSON 文本的解析结果

        Raises:
            ValueError: 当文件内容不是有效 JSON 或无法解码时

        """
        text = file_content.decode("utf-8")
        data = json.loads(text)
        normalized = json.dumps(data, ensure_ascii=False, indent=2)
        return ParseResult(text=normalized, media=[])
