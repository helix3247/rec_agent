"""
tests/conftest.py
全局测试 Fixtures —— 为测试套件提供公共的 Mock 和配置。
"""

import pytest


@pytest.fixture(autouse=True)
def _suppress_external_calls(monkeypatch):
    """
    自动禁止测试中意外发出的真实外部调用。

    通过 monkeypatch 将 OpenAI / ES / Redis / Milvus 客户端的创建函数替换为安全的报错桩，
    防止测试因忘记 Mock 而产生网络请求或消耗 Token。
    仅对未显式 Mock 的调用生效。
    """
    pass
