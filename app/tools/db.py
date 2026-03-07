"""
app/tools/db.py
数据库工具 —— 封装 MySQL 查询逻辑，提供用户画像、收藏夹、订单查询等能力。
"""

import json
from typing import Optional

import pymysql
import pymysql.cursors

from app.core.config import settings
from app.core.logger import get_logger

_logger = get_logger(agent_name="DBTool")


def _get_connection() -> pymysql.connections.Connection:
    """获取 MySQL 连接。"""
    mysql_cfg = settings.mysql
    return pymysql.connect(
        host=mysql_cfg.mysql_host,
        port=mysql_cfg.mysql_port,
        user=mysql_cfg.mysql_user,
        password=mysql_cfg.mysql_password,
        database=mysql_cfg.mysql_database,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )


def get_user_profile(user_id: str) -> Optional[dict]:
    """
    获取用户画像：基础信息 + 历史行为偏好。

    Returns:
        {
            "user_id", "name", "age", "gender", "budget_level",
            "style_preference": [...],
            "category_interest": [...],
            "purchase_history": [{"product_id", "name", "category", "brand", "price"}],
            "liked_brands": [...],
            "liked_categories": [...],
            "price_range": {"min", "max", "avg"},
        }
        若用户不存在则返回 None。
    """
    if not user_id:
        return None

    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            # 基础信息
            cur.execute("SELECT * FROM users WHERE user_id = %s", (user_id,))
            user = cur.fetchone()
            if not user:
                _logger.info("用户不存在 | user_id={}", user_id)
                return None

            profile = {
                "user_id": user["user_id"],
                "name": user["name"],
                "age": user["age"],
                "gender": user["gender"],
                "budget_level": user["budget_level"],
                "style_preference": _parse_json_field(user.get("style_pref", "[]")),
                "category_interest": _parse_json_field(user.get("category_interest", "[]")),
            }

            # 购买历史（最近 20 单）
            cur.execute("""
                SELECT p.id AS product_id, p.name, p.category, p.brand, p.price
                FROM orders o
                JOIN products p ON o.product_id = p.id
                WHERE o.user_id = %s AND o.status != 'cancelled'
                ORDER BY o.created_at DESC
                LIMIT 20
            """, (user_id,))
            purchases = cur.fetchall()
            profile["purchase_history"] = [dict(p) for p in purchases]

            # 从购买记录中提取偏好
            if purchases:
                brands = list({p["brand"] for p in purchases})
                categories = list({p["category"] for p in purchases})
                prices = [float(p["price"]) for p in purchases]
                profile["liked_brands"] = brands
                profile["liked_categories"] = categories
                profile["price_range"] = {
                    "min": min(prices),
                    "max": max(prices),
                    "avg": round(sum(prices) / len(prices), 2),
                }
            else:
                profile["liked_brands"] = []
                profile["liked_categories"] = []
                profile["price_range"] = {"min": 0, "max": 0, "avg": 0}

            # 点赞 / 收藏行为统计
            cur.execute("""
                SELECT p.brand, p.category, COUNT(*) AS cnt
                FROM interactions i
                JOIN products p ON i.product_id = p.id
                WHERE i.user_id = %s AND i.action IN ('like', 'cart')
                GROUP BY p.brand, p.category
                ORDER BY cnt DESC
                LIMIT 10
            """, (user_id,))
            fav_stats = cur.fetchall()
            if fav_stats:
                fav_brands = list({row["brand"] for row in fav_stats})
                fav_cats = list({row["category"] for row in fav_stats})
                profile["liked_brands"] = list(set(profile["liked_brands"] + fav_brands))
                profile["liked_categories"] = list(set(profile["liked_categories"] + fav_cats))

            _logger.info("用户画像获取成功 | user_id={} | purchases={}", user_id, len(purchases))
            return profile
    except Exception as e:
        _logger.error("用户画像查询失败 | user_id={} | error={}", user_id, str(e))
        return None
    finally:
        conn.close()


def list_favorites(user_id: str, limit: int = 20) -> list[dict]:
    """
    获取用户收藏夹列表。

    Returns:
        [{product_id, name, category, brand, price}]
    """
    if not user_id:
        return []

    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT p.id AS product_id, p.name, p.category, p.brand, p.price
                FROM favorites f
                JOIN products p ON f.product_id = p.id
                WHERE f.user_id = %s
                ORDER BY f.created_at DESC
                LIMIT %s
            """, (user_id, limit))
            rows = cur.fetchall()
            _logger.info("收藏夹查询成功 | user_id={} | count={}", user_id, len(rows))
            return [dict(r) for r in rows]
    except Exception as e:
        _logger.error("收藏夹查询失败 | user_id={} | error={}", user_id, str(e))
        return []
    finally:
        conn.close()


def get_favorite_by_id(user_id: str, product_id: str) -> Optional[dict]:
    """
    查询用户收藏夹中的特定商品。

    Returns:
        {product_id, name, category, brand, price, description} 或 None。
    """
    if not user_id or not product_id:
        return None

    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT p.id AS product_id, p.name, p.category, p.brand,
                       p.price, p.description, p.specs, p.tags
                FROM favorites f
                JOIN products p ON f.product_id = p.id
                WHERE f.user_id = %s AND f.product_id = %s
            """, (user_id, product_id))
            row = cur.fetchone()
            if row:
                result = dict(row)
                result["specs"] = _parse_json_field(result.get("specs", "{}"))
                result["tags"] = _parse_json_field(result.get("tags", "[]"))
                return result
            return None
    except Exception as e:
        _logger.error("收藏夹商品查询失败 | error={}", str(e))
        return None
    finally:
        conn.close()


def get_product_by_id(product_id: str) -> Optional[dict]:
    """根据商品 ID 查询商品详情。"""
    if not product_id:
        return None

    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM products WHERE id = %s", (product_id,))
            row = cur.fetchone()
            if row:
                result = dict(row)
                result["specs"] = _parse_json_field(result.get("specs", "{}"))
                result["tags"] = _parse_json_field(result.get("tags", "[]"))
                return result
            return None
    except Exception as e:
        _logger.error("商品查询失败 | product_id={} | error={}", product_id, str(e))
        return None
    finally:
        conn.close()


def query_order_status(user_id: str, order_id: Optional[str] = None) -> list[dict]:
    """
    查询用户订单状态。

    Args:
        user_id: 用户 ID。
        order_id: 订单 ID（可选，为空则返回最近订单）。

    Returns:
        [{order_id, product_id, product_name, quantity, total_price, status}]
    """
    if not user_id:
        return []

    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            if order_id:
                cur.execute("""
                    SELECT o.order_id, o.product_id, p.name AS product_name,
                           o.quantity, o.total_price, o.status, o.created_at
                    FROM orders o
                    JOIN products p ON o.product_id = p.id
                    WHERE o.user_id = %s AND o.order_id = %s
                """, (user_id, order_id))
            else:
                cur.execute("""
                    SELECT o.order_id, o.product_id, p.name AS product_name,
                           o.quantity, o.total_price, o.status, o.created_at
                    FROM orders o
                    JOIN products p ON o.product_id = p.id
                    WHERE o.user_id = %s
                    ORDER BY o.created_at DESC
                    LIMIT 5
                """, (user_id,))
            rows = cur.fetchall()
            results = []
            for r in rows:
                item = dict(r)
                if item.get("created_at"):
                    item["created_at"] = str(item["created_at"])
                item["total_price"] = float(item["total_price"])
                results.append(item)
            return results
    except Exception as e:
        _logger.error("订单查询失败 | user_id={} | error={}", user_id, str(e))
        return []
    finally:
        conn.close()


def _parse_json_field(value) -> list | dict:
    """安全解析 JSON 字段。"""
    if isinstance(value, (list, dict)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return []
    return []
