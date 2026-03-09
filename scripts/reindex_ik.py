"""
scripts/reindex_ik.py
将 product_index 迁移到使用 IK 中文分词 + 同义词的新 mapping。

使用方式:
    conda activate rec_agent
    python scripts/reindex_ik.py
"""

from elasticsearch import Elasticsearch

ES_HOST = "http://127.0.0.1:29200"
ES_AUTH = ("elastic", "changeme")
INDEX = "product_index"
TMP_INDEX = f"{INDEX}_ik_tmp"

SETTINGS = {
    "settings": {
        "analysis": {
            "filter": {
                "synonym_filter": {
                    "type": "synonym",
                    "synonyms_path": "analysis/synonym.txt",
                    "updateable": True,
                }
            },
            "analyzer": {
                "ik_max_index": {
                    "type": "custom",
                    "tokenizer": "ik_max_word",
                    "filter": ["lowercase"],
                },
                "ik_smart_search": {
                    "type": "custom",
                    "tokenizer": "ik_smart",
                    "filter": ["lowercase", "synonym_filter"],
                },
            },
        }
    },
    "mappings": {
        "properties": {
            "product_id": {"type": "keyword"},
            "name": {
                "type": "text",
                "analyzer": "ik_max_index",
                "search_analyzer": "ik_smart_search",
            },
            "category": {"type": "keyword"},
            "brand": {"type": "keyword"},
            "price": {"type": "float"},
            "tags": {"type": "keyword"},
            "description": {
                "type": "text",
                "analyzer": "ik_max_index",
                "search_analyzer": "ik_smart_search",
            },
            "embedding": {
                "type": "dense_vector",
                "dims": 3072,
                "index": True,
                "similarity": "cosine",
            },
        }
    },
}


def main():
    es = Elasticsearch(ES_HOST, basic_auth=ES_AUTH)
    print(f"Connected to ES: {es.info()['version']['number']}")

    # Step 1: 清理可能残留的临时索引
    if es.indices.exists(index=TMP_INDEX):
        es.indices.delete(index=TMP_INDEX)
        print(f"[1/5] 已清理残留临时索引 {TMP_INDEX}")

    # Step 2: 创建临时索引（新 mapping）
    es.indices.create(index=TMP_INDEX, body=SETTINGS)
    print(f"[2/5] 临时索引 {TMP_INDEX} 已创建（IK 分词 + 同义词）")

    # Step 3: 将旧索引数据迁移到临时索引
    resp = es.reindex(
        body={"source": {"index": INDEX}, "dest": {"index": TMP_INDEX}},
        wait_for_completion=True,
    )
    created = resp.get("created", 0)
    print(f"[3/5] Reindex {INDEX} → {TMP_INDEX} 完成 | docs={created}")

    # Step 4: 删除旧索引，用新 mapping 重建
    es.indices.delete(index=INDEX)
    es.indices.create(index=INDEX, body=SETTINGS)
    print(f"[4/5] 旧索引已删除，新索引 {INDEX} 已创建")

    # Step 5: 迁移回正式索引
    resp2 = es.reindex(
        body={"source": {"index": TMP_INDEX}, "dest": {"index": INDEX}},
        wait_for_completion=True,
    )
    created2 = resp2.get("created", 0)
    es.indices.delete(index=TMP_INDEX)
    print(f"[5/5] Reindex {TMP_INDEX} → {INDEX} 完成 | docs={created2} | 临时索引已清理")

    # 验证结果
    count = es.count(index=INDEX)["count"]
    mapping = es.indices.get_mapping(index=INDEX)
    name_cfg = mapping[INDEX]["mappings"]["properties"]["name"]
    desc_cfg = mapping[INDEX]["mappings"]["properties"]["description"]

    print(f"\n{'='*50}")
    print(f"  Reindex 完成")
    print(f"  文档数: {count}")
    print(f"  name.analyzer: {name_cfg.get('analyzer', 'N/A')}")
    print(f"  name.search_analyzer: {name_cfg.get('search_analyzer', 'N/A')}")
    print(f"  description.analyzer: {desc_cfg.get('analyzer', 'N/A')}")
    print(f"{'='*50}")

    # 测试 IK 分词效果
    analyze_resp = es.indices.analyze(
        index=INDEX,
        body={"analyzer": "ik_smart_search", "text": "我想买一件红色的卫衣"},
    )
    tokens = [t["token"] for t in analyze_resp["tokens"]]
    print(f"\n  IK 分词测试: '我想买一件红色的卫衣'")
    print(f"  分词结果: {tokens}")


if __name__ == "__main__":
    main()
