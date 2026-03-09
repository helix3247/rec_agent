"""
app/tools/personalization.py
个性化排序模块 —— 基于用户画像对检索结果进行重排序。

排序因子:
  - 品牌偏好：用户历史购买/收藏的品牌加权
  - 价格区间偏好：与用户消费水平匹配度
  - 品类偏好：用户感兴趣品类加权
  - 品类互补性：与历史购买品类的互补关系

设计为可插拔模块，后续可替换为更复杂的排序模型。
"""

from typing import Optional

from app.core.logger import get_logger

_logger = get_logger(agent_name="Personalization")

# 品类互补关系映射（历史购买品类 -> 互补品类）
_COMPLEMENTARY_CATEGORIES = {
    "相机": ["镜头", "三脚架", "存储卡", "相机包"],
    "手机": ["手机壳", "耳机", "充电器", "手机膜"],
    "笔记本电脑": ["鼠标", "键盘", "显示器", "电脑包"],
    "耳机": ["耳机盒", "音箱"],
    "外套": ["裤子", "鞋子", "围巾", "帽子"],
    "上衣": ["裤子", "鞋子", "配饰"],
    "裤子": ["鞋子", "腰带", "上衣"],
    "鞋子": ["袜子", "鞋垫"],
    "手表": ["表带", "首饰"],
}

# 预算等级到价格范围映射
_BUDGET_LEVEL_RANGES = {
    "low":  (0, 500),
    "mid":  (300, 3000),
    "high": (2000, float("inf")),
}


def rerank_by_user_profile(
    products: list[dict],
    user_profile: Optional[dict],
) -> list[dict]:
    """
    基于用户画像对商品列表进行个性化重排序。

    Args:
        products: ES 检索结果列表，每个 dict 至少包含 product_id, name, category, brand, price, score。
        user_profile: 用户画像。为 None 时退化为默认排序（按原始 score）。

    Returns:
        重排序后的商品列表。
    """
    if not products:
        return products

    if not user_profile:
        _logger.info("无用户画像，使用默认排序")
        return list(products)

    liked_brands = set(user_profile.get("liked_brands", []))
    liked_categories = set(user_profile.get("liked_categories", []))
    category_interests = set(user_profile.get("category_interest", []))
    budget_level = user_profile.get("budget_level", "mid")
    price_range = user_profile.get("price_range", {})

    purchased_categories = set()
    for p in user_profile.get("purchase_history", []):
        purchased_categories.add(p.get("category", ""))

    scored: list[tuple[float, dict]] = []
    for product in products:
        original_score = product.get("score", 0)
        boost = 0.0

        brand = product.get("brand", "")
        category = product.get("category", "")
        price = product.get("price", 0)

        if brand in liked_brands:
            boost += 2.0

        if category in liked_categories or category in category_interests:
            boost += 1.5

        budget_range = _BUDGET_LEVEL_RANGES.get(budget_level, (0, float("inf")))
        if budget_range[0] <= price <= budget_range[1]:
            boost += 1.0

        avg_price = price_range.get("avg", 0)
        if avg_price > 0 and price > 0:
            price_ratio = min(price, avg_price) / max(price, avg_price)
            boost += price_ratio * 0.5

        for purchased_cat in purchased_categories:
            complementary = _COMPLEMENTARY_CATEGORIES.get(purchased_cat, [])
            if category in complementary:
                boost += 1.0
                break

        final_score = round(original_score + boost, 4)
        scored.append((final_score, product))

    scored.sort(key=lambda x: x[0], reverse=True)

    _logger.info(
        "个性化排序完成 | products={} | brand_prefs={} | budget_level={}",
        len(products), len(liked_brands), budget_level,
    )
    return [product for _, product in scored]
