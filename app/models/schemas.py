"""
app/models/schemas.py
API 请求 / 响应的 Pydantic 模型定义。
"""

from typing import Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """POST /chat 请求体"""
    query: str = Field(..., description="用户输入的自然语言查询")
    thread_id: Optional[str] = Field(None, description="会话 ID，为空则创建新会话")
    user_id: Optional[str] = Field(None, description="用户 ID，用于个性化推荐")
    selected_product_id: Optional[str] = Field(None, description="用户选中的商品 ID")


class CandidateItem(BaseModel):
    """候选商品"""
    product_id: str = ""
    title: str = ""
    price: float = 0.0
    reason: str = ""


class ChatResponse(BaseModel):
    """POST /chat 响应体"""
    response: str = Field("", description="Agent 的回复文本")
    trace_id: str = Field("", description="链路追踪 ID")
    thread_id: str = Field("", description="会话 ID")
    suggested_questions: list[str] = Field(default_factory=list, description="推荐的后续问题")
    candidates: list[CandidateItem] = Field(default_factory=list, description="候选商品列表")
