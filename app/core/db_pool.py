"""
app/core/db_pool.py
MySQL 连接池管理模块。

提供全局连接池的初始化、获取连接、释放连接和关闭池。
应用启动时调用 init_pool()，关闭时调用 close_pool()。
"""

from contextlib import asynccontextmanager
from typing import Optional

import aiomysql

from app.core.config import settings
from app.core.logger import get_logger

_logger = get_logger(agent_name="DBPool")


class DBPool:
    """MySQL 异步连接池管理器。"""

    _pool: Optional[aiomysql.Pool] = None

    @classmethod
    async def init_pool(
        cls,
        minsize: int = 5,
        maxsize: int = 20,
        connect_timeout: int = 10,
    ) -> None:
        """
        初始化连接池，应用启动时调用。

        Args:
            minsize: 连接池最小连接数。
            maxsize: 连接池最大连接数。
            connect_timeout: 单个连接的超时秒数。
        """
        if cls._pool is not None:
            _logger.warning("连接池已初始化，跳过重复调用")
            return

        mysql_cfg = settings.mysql
        try:
            cls._pool = await aiomysql.create_pool(
                host=mysql_cfg.mysql_host,
                port=mysql_cfg.mysql_port,
                user=mysql_cfg.mysql_user,
                password=mysql_cfg.mysql_password,
                db=mysql_cfg.mysql_database,
                minsize=minsize,
                maxsize=maxsize,
                connect_timeout=connect_timeout,
                charset="utf8mb4",
                autocommit=True,
                cursorclass=aiomysql.DictCursor,
            )
            _logger.info(
                "MySQL 连接池初始化成功 | host={}:{} | db={} | pool=[{}, {}]",
                mysql_cfg.mysql_host, mysql_cfg.mysql_port,
                mysql_cfg.mysql_database, minsize, maxsize,
            )
        except Exception as e:
            _logger.error("MySQL 连接池初始化失败 | error={}", str(e))
            raise

    @classmethod
    async def close_pool(cls) -> None:
        """关闭连接池，应用关闭时调用。"""
        if cls._pool is not None:
            cls._pool.close()
            await cls._pool.wait_closed()
            cls._pool = None
            _logger.info("MySQL 连接池已关闭")

    @classmethod
    def get_pool(cls) -> Optional[aiomysql.Pool]:
        """获取连接池实例。"""
        return cls._pool

    @classmethod
    @asynccontextmanager
    async def acquire(cls):
        """
        以上下文管理器方式获取连接，用完自动释放回池。

        Usage:
            async with DBPool.acquire() as conn:
                async with conn.cursor() as cur:
                    await cur.execute("SELECT ...")
                    rows = await cur.fetchall()
        """
        if cls._pool is None:
            raise RuntimeError("MySQL 连接池未初始化，请先调用 DBPool.init_pool()")

        conn = await cls._pool.acquire()
        try:
            yield conn
        finally:
            cls._pool.release(conn)

    @classmethod
    async def execute_query(
        cls,
        sql: str,
        args: tuple = (),
        fetch_one: bool = False,
    ) -> list[dict] | dict | None:
        """
        便捷查询方法：自动获取连接、执行、释放。

        Args:
            sql: SQL 语句。
            args: SQL 参数。
            fetch_one: True 返回单条记录，False 返回列表。

        Returns:
            查询结果。
        """
        async with cls.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(sql, args)
                if fetch_one:
                    row = await cur.fetchone()
                    return dict(row) if row else None
                rows = await cur.fetchall()
                return [dict(r) for r in rows]

    @classmethod
    async def pool_status(cls) -> dict:
        """返回连接池当前状态。"""
        if cls._pool is None:
            return {"initialized": False}
        return {
            "initialized": True,
            "size": cls._pool.size,
            "free_size": cls._pool.freesize,
            "min_size": cls._pool.minsize,
            "max_size": cls._pool.maxsize,
        }
