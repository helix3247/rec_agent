"""
app/models/intent.py
意图识别结构化输出 Schema —— 配合 llm.with_structured_output 使用。
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class IntentResult(BaseModel):
    """LLM 意图识别的结构化输出。"""

    intent: Literal["search", "outfit", "qa", "chat", "compare", "plan"] = Field(
        ...,
        description="用户意图分类: search(商品搜索), outfit(穿搭推荐), qa(商品问答), "
                    "chat(闲聊), compare(商品对比), plan(复杂任务规划)",
    )
    budget: Optional[str] = Field(
        None,
        description="预算范围，如 '5000元以内'、'3000-5000'",
    )
    category: Optional[str] = Field(
        None,
        description="商品品类，如 '相机'、'手机'、'外套'",
    )
    scenario: Optional[str] = Field(
        None,
        description="使用场景，如 '旅行'、'通勤'、'约会'",
    )
    style: Optional[str] = Field(
        None,
        description="风格偏好，如 '复古'、'简约'、'运动'",
    )
    must_have: Optional[str] = Field(
        None,
        description="必须具备的特征，如 '防水'、'轻便'、'高像素'",
    )
