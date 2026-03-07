"""
scripts/init_mysql.py
初始化 MySQL 表结构并将 mock_data.json 中的用户和商品数据批量写入。

执行方式:
    python scripts/init_mysql.py
    python scripts/init_mysql.py --drop-existing   # 重建时先删旧表
"""

import argparse
import json
import sys

import pymysql
import pymysql.cursors

from config import MOCK_DATA_FILE, MYSQL_CONFIG

# ─────────────────────────── DDL 定义 ───────────────────────────

DDL_USERS = """
CREATE TABLE IF NOT EXISTS users (
    user_id       VARCHAR(36)  NOT NULL PRIMARY KEY COMMENT '用户UUID',
    name          VARCHAR(64)  NOT NULL COMMENT '用户名',
    age           TINYINT      NOT NULL COMMENT '年龄',
    gender        VARCHAR(10)  NOT NULL COMMENT '性别 male/female',
    budget_level  VARCHAR(10)  NOT NULL COMMENT '预算档次 high/mid/low',
    style_pref    JSON         NOT NULL COMMENT '风格偏好标签列表',
    category_interest JSON     NOT NULL COMMENT '感兴趣品类',
    created_at    TIMESTAMP    DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='用户画像';
"""

DDL_PRODUCTS = """
CREATE TABLE IF NOT EXISTS products (
    id          VARCHAR(36)     NOT NULL PRIMARY KEY COMMENT '商品UUID',
    name        VARCHAR(255)    NOT NULL COMMENT '商品名称',
    category    VARCHAR(64)     NOT NULL COMMENT '品类',
    price       DECIMAL(10, 2)  NOT NULL COMMENT '价格（元）',
    brand       VARCHAR(64)     NOT NULL COMMENT '品牌',
    specs       JSON            NOT NULL COMMENT '商品规格（像素/材质等）',
    tags        JSON            NOT NULL COMMENT '标签列表',
    description TEXT            NOT NULL COMMENT '商品描述（用于Embedding）',
    created_at  TIMESTAMP       DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='商品基础信息';
"""

DDL_INTERACTIONS = """
CREATE TABLE IF NOT EXISTS interactions (
    id          BIGINT      NOT NULL AUTO_INCREMENT PRIMARY KEY,
    user_id     VARCHAR(36) NOT NULL COMMENT '用户ID',
    product_id  VARCHAR(36) NOT NULL COMMENT '商品ID',
    action      VARCHAR(20) NOT NULL COMMENT '行为类型: view/like/cart/purchase',
    session_id  VARCHAR(64) COMMENT '会话ID',
    created_at  TIMESTAMP   DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_user (user_id),
    INDEX idx_product (product_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='用户行为交互日志';
"""

DDL_ORDERS = """
CREATE TABLE IF NOT EXISTS orders (
    order_id    VARCHAR(36)    NOT NULL PRIMARY KEY,
    user_id     VARCHAR(36)    NOT NULL COMMENT '用户ID',
    product_id  VARCHAR(36)    NOT NULL COMMENT '商品ID',
    quantity    INT            NOT NULL DEFAULT 1,
    total_price DECIMAL(10, 2) NOT NULL COMMENT '实付金额',
    status      VARCHAR(20)    NOT NULL DEFAULT 'pending'
                               COMMENT '订单状态: pending/paid/shipped/done/cancelled',
    created_at  TIMESTAMP      DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_user (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='订单表';
"""

DROP_ORDER = ["orders", "interactions", "products", "users"]
ALL_DDL = [DDL_USERS, DDL_PRODUCTS, DDL_INTERACTIONS, DDL_ORDERS]


# ─────────────────────────── 工具函数 ───────────────────────────

def get_connection() -> pymysql.connections.Connection:
    cfg = MYSQL_CONFIG.copy()
    cfg.pop("database", None)
    return pymysql.connect(
        host=MYSQL_CONFIG["host"],
        port=MYSQL_CONFIG["port"],
        user=MYSQL_CONFIG["user"],
        password=MYSQL_CONFIG["password"],
        charset=MYSQL_CONFIG["charset"],
        cursorclass=pymysql.cursors.DictCursor,
    )


def ensure_database(conn: pymysql.connections.Connection) -> None:
    db = MYSQL_CONFIG["database"]
    with conn.cursor() as cursor:
        cursor.execute(
            f"CREATE DATABASE IF NOT EXISTS `{db}` "
            f"CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
        )
    conn.commit()
    conn.select_db(db)
    print(f"  数据库 `{db}` 就绪")


def drop_tables(conn: pymysql.connections.Connection) -> None:
    with conn.cursor() as cursor:
        cursor.execute("SET FOREIGN_KEY_CHECKS = 0;")
        for table in DROP_ORDER:
            cursor.execute(f"DROP TABLE IF EXISTS `{table}`;")
            print(f"  已删除表 `{table}`")
        cursor.execute("SET FOREIGN_KEY_CHECKS = 1;")
    conn.commit()


def create_tables(conn: pymysql.connections.Connection) -> None:
    with conn.cursor() as cursor:
        for ddl in ALL_DDL:
            cursor.execute(ddl)
    conn.commit()
    print("  表结构创建完成: users / products / interactions / orders")


def insert_users(conn: pymysql.connections.Connection, users: list) -> int:
    sql = """
        INSERT IGNORE INTO users
            (user_id, name, age, gender, budget_level, style_pref, category_interest)
        VALUES
            (%(user_id)s, %(name)s, %(age)s, %(gender)s,
             %(budget_level)s, %(style_pref)s, %(category_interest)s)
    """
    rows = [
        {
            "user_id": u["user_id"],
            "name": u["name"],
            "age": u["age"],
            "gender": u["gender"],
            "budget_level": u["budget_level"],
            "style_pref": json.dumps(u["style_preference"], ensure_ascii=False),
            "category_interest": json.dumps(u["category_interest"], ensure_ascii=False),
        }
        for u in users
    ]
    with conn.cursor() as cursor:
        cursor.executemany(sql, rows)
    conn.commit()
    return len(rows)


def insert_products(conn: pymysql.connections.Connection, products: list) -> int:
    sql = """
        INSERT IGNORE INTO products
            (id, name, category, price, brand, specs, tags, description)
        VALUES
            (%(id)s, %(name)s, %(category)s, %(price)s,
             %(brand)s, %(specs)s, %(tags)s, %(description)s)
    """
    rows = [
        {
            "id": p["id"],
            "name": p["name"],
            "category": p["category"],
            "price": p["price"],
            "brand": p["brand"],
            "specs": json.dumps(p["specs"], ensure_ascii=False),
            "tags": json.dumps(p["tags"], ensure_ascii=False),
            "description": p["description"],
        }
        for p in products
    ]
    with conn.cursor() as cursor:
        cursor.executemany(sql, rows)
    conn.commit()
    return len(rows)


def insert_interactions(conn: pymysql.connections.Connection, interactions: list) -> int:
    if not interactions:
        return 0
    sql = """
        INSERT INTO interactions
            (user_id, product_id, action, session_id)
        VALUES
            (%(user_id)s, %(product_id)s, %(action)s, %(session_id)s)
    """
    with conn.cursor() as cursor:
        cursor.executemany(sql, interactions)
    conn.commit()
    return len(interactions)


def insert_orders(conn: pymysql.connections.Connection, orders: list) -> int:
    if not orders:
        return 0
    sql = """
        INSERT INTO orders
            (order_id, user_id, product_id, quantity, total_price, status)
        VALUES
            (%(order_id)s, %(user_id)s, %(product_id)s,
             %(quantity)s, %(total_price)s, %(status)s)
    """
    with conn.cursor() as cursor:
        cursor.executemany(sql, orders)
    conn.commit()
    return len(orders)


# ─────────────────────────── 主流程 ───────────────────────────

def main(drop_existing: bool) -> None:
    if not MOCK_DATA_FILE.exists():
        print(f"[错误] 未找到数据文件: {MOCK_DATA_FILE}")
        print("请先运行 python scripts/generate_mock_data.py 生成数据。")
        sys.exit(1)

    print(f"[1/4] 读取数据文件: {MOCK_DATA_FILE}")
    with open(MOCK_DATA_FILE, "r", encoding="utf-8") as f:
        mock_data = json.load(f)
    users = mock_data.get("users", [])
    products = mock_data.get("products", [])
    interactions = mock_data.get("interactions", [])
    orders = mock_data.get("orders", [])
    print(
        f"  用户 {len(users)} 条，商品 {len(products)} 条，"
        f"交互 {len(interactions)} 条，订单 {len(orders)} 条"
    )

    print("[2/4] 连接 MySQL ...")
    conn = get_connection()
    ensure_database(conn)

    if drop_existing:
        print("[2b] 删除旧表...")
        drop_tables(conn)

    print("[3/4] 初始化表结构...")
    create_tables(conn)

    print("[4/4] 写入数据...")
    n_users = insert_users(conn, users)
    print(f"  用户写入 {n_users} 条")
    n_products = insert_products(conn, products)
    print(f"  商品写入 {n_products} 条")
    n_interactions = insert_interactions(conn, interactions)
    print(f"  交互写入 {n_interactions} 条")
    n_orders = insert_orders(conn, orders)
    print(f"  订单写入 {n_orders} 条")

    conn.close()
    print("✓ MySQL 初始化完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="初始化 MySQL 并写入 Mock 数据")
    parser.add_argument(
        "--drop-existing",
        action="store_true",
        help="重建表结构前先删除已有表（用于重置数据）",
    )
    args = parser.parse_args()
    main(drop_existing=args.drop_existing)
