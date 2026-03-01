"""
scripts/generate_mock_data.py
生成包含用户、商品、评论的黄金源数据，输出到 data/mock_data.json。

执行方式:
    python scripts/generate_mock_data.py
    python scripts/generate_mock_data.py --num-products 60 --seed 0
"""

import argparse
import json
import random
import uuid
from pathlib import Path

from config import DATA_CONFIG, DATA_DIR, MOCK_DATA_FILE

# ─────────────────────────── 数码相机素材库 ───────────────────────────
CAMERA_BRANDS = ["Sony", "Canon", "Nikon", "Fujifilm", "Panasonic", "Leica", "OM System"]
CAMERA_SERIES = ["A7M4", "A6700", "ZV-E10", "EOS R8", "EOS R50", "Z5II", "Z30",
                 "X-T5", "X-S20", "X100VI", "GH7", "G9II", "M11", "OM-5"]
CAMERA_ADJECTIVES = ["全画幅微单", "APS-C微单", "入门微单", "旗舰微单",
                     "Vlog相机", "复古旁轴", "专业无反"]
CAMERA_TAGS_POOL = ["全画幅", "APS-C", "4K视频", "防抖", "复古风格",
                    "轻便", "高像素", "高速连拍", "夜景", "Vlog"]

CAMERA_SPECS_TEMPLATES = [
    {"像素": "{px}MP", "传感器": "{sensor}", "防抖": "{ois}", "视频": "{video}"},
]
CAMERA_SENSOR_OPTIONS = ["全画幅BSI-CMOS", "APS-C BSI-CMOS", "M4/3 CMOS"]
CAMERA_OIS_OPTIONS = ["5轴机身防抖", "光学防抖", "无"]
CAMERA_VIDEO_OPTIONS = ["4K 60fps", "4K 30fps", "6K 30fps", "4K 120fps"]

CAMERA_DESC_TEMPLATES = [
    "{brand} {model} 搭载 {px}MP {sensor}，{ois}，支持 {video} 视频录制。"
    "适合{adj}用户，{tag1}表现出色，深受{use_case}人群喜爱。",
    "专业级{adj}，{brand} {model}，{px}百万像素，配备{sensor}，"
    "{ois}设计，{tag1}与{tag2}兼顾，是{use_case}的理想之选。",
]
CAMERA_USE_CASES = ["旅行摄影", "街头扫街", "人像拍摄", "风光摄影", "婚礼记录", "Vlog创作"]

CAMERA_REVIEW_TEMPLATES = [
    "这款{brand}的{feature}表现非常出色，在{scenario}场景下几乎没有噪点，但在{weakness}时稍显不足。",
    "用了两个月，{feature}让我非常满意，{tag}风格拍出来的照片很有质感，续航是个小短板。",
    "对比过{compare}，最终选择了{brand} {model}，主要是看中了它的{feature}和体积优势。",
    "说明书里提到的{feature}参数确实名副其实，{scenario}实测下来信噪比很好，强烈推荐。",
    "{brand} {model} 上手难度适中，{feature}功能操作逻辑清晰，非常适合{use_case}新手入门。",
]

# ─────────────────────────── 男装素材库 ───────────────────────────
CLOTHING_BRANDS = ["Uniqlo", "HM", "ZARA", "Only & Sons", "Jack Jones",
                   "雅戈尔", "海澜之家", "GXG", "太平鸟", "Calvin Klein"]
CLOTHING_TYPES = ["休闲衬衫", "商务衬衫", "牛仔裤", "休闲裤",
                  "连帽卫衣", "圆领T恤", "Polo衫", "西装外套",
                  "羽绒服", "风衣", "针织衫", "运动套装"]
CLOTHING_ADJECTIVES = ["宽松版型", "修身版型", "直筒版型", "oversize"]
CLOTHING_TAGS_POOL = ["商务", "休闲", "复古", "潮流", "运动", "日系", "简约", "街头"]

CLOTHING_MATERIALS = ["100%纯棉", "棉麻混纺", "聚酯纤维", "羊毛混纺", "莫代尔棉"]
CLOTHING_COLORS = ["黑色", "白色", "藏青", "卡其色", "灰色", "深蓝", "米白"]

CLOTHING_DESC_TEMPLATES = [
    "{brand} {adj}{type}，采用{material}面料，{color}配色，{tag1}风格，"
    "适合{occasion}场合穿搭，版型{fit}，穿着舒适透气。",
    "{adj}设计的{brand}{type}，{material}材质手感细腻，{tag1}与{tag2}风格融合，"
    "{color}系配色百搭耐看，是{season}必备单品。",
]
CLOTHING_OCCASIONS = ["日常通勤", "休闲出行", "商务会议", "约会聚餐", "户外运动"]
CLOTHING_SEASONS = ["春夏", "秋冬", "四季", "夏季", "冬季"]

CLOTHING_REVIEW_TEMPLATES = [
    "这件{brand}的{type}面料{material}，上身{tag}感十足，{occasion}穿刚刚好，洗了几次没缩水。",
    "版型是{adj}的，身高{height}cm体重{weight}kg穿{size}码，腰围和肩宽都挺合适。",
    "颜色和图片基本一致，{color}很耐看，{occasion}搭配{match}显得很有品味。",
    "做工不错，缝线均匀，{material}面料透气性好，{season}穿着非常舒适。",
    "{brand}的品控一如既往稳定，{type}穿了一整个{season}，弹性和颜色都没怎么变。",
]

# ─────────────────────────── 用户画像模板 ───────────────────────────
USER_PROFILES = [
    {"name": "张明", "age": 28, "gender": "male", "budget_level": "mid",
     "style_preference": ["休闲", "日系"], "category_interest": ["男装", "数码相机"]},
    {"name": "李浩", "age": 35, "gender": "male", "budget_level": "high",
     "style_preference": ["商务", "简约"], "category_interest": ["数码相机", "男装"]},
    {"name": "王磊", "age": 22, "gender": "male", "budget_level": "low",
     "style_preference": ["街头", "潮流"], "category_interest": ["男装"]},
    {"name": "陈晨", "age": 30, "gender": "male", "budget_level": "mid",
     "style_preference": ["复古", "休闲"], "category_interest": ["数码相机"]},
    {"name": "刘阳", "age": 45, "gender": "male", "budget_level": "high",
     "style_preference": ["商务"], "category_interest": ["男装", "数码相机"]},
    {"name": "赵雪", "age": 26, "gender": "female", "budget_level": "mid",
     "style_preference": ["日系", "简约"], "category_interest": ["数码相机"]},
    {"name": "孙嘉", "age": 32, "gender": "female", "budget_level": "high",
     "style_preference": ["潮流", "街头"], "category_interest": ["男装", "数码相机"]},
    {"name": "周峰", "age": 19, "gender": "male", "budget_level": "low",
     "style_preference": ["运动", "休闲"], "category_interest": ["男装"]},
    {"name": "吴凯", "age": 38, "gender": "male", "budget_level": "high",
     "style_preference": ["商务", "简约"], "category_interest": ["数码相机"]},
    {"name": "郑宇", "age": 24, "gender": "male", "budget_level": "mid",
     "style_preference": ["复古", "潮流"], "category_interest": ["男装", "数码相机"]},
]


# ─────────────────────────── 生成函数 ───────────────────────────

def _rand_tags(pool: list, k: int = 3) -> list:
    return random.sample(pool, min(k, len(pool)))


def _generate_camera(product_id: str) -> dict:
    brand = random.choice(CAMERA_BRANDS)
    model = random.choice(CAMERA_SERIES)
    adj = random.choice(CAMERA_ADJECTIVES)
    px = random.choice([12, 24, 26, 33, 61])
    sensor = random.choice(CAMERA_SENSOR_OPTIONS)
    ois = random.choice(CAMERA_OIS_OPTIONS)
    video = random.choice(CAMERA_VIDEO_OPTIONS)
    tags = _rand_tags(CAMERA_TAGS_POOL, 3)
    tag1, tag2 = tags[0], tags[1] if len(tags) > 1 else tags[0]
    use_case = random.choice(CAMERA_USE_CASES)

    specs = {"像素": f"{px}MP", "传感器": sensor, "防抖": ois, "视频": video}
    desc_tmpl = random.choice(CAMERA_DESC_TEMPLATES)
    description = desc_tmpl.format(
        brand=brand, model=model, px=px, sensor=sensor,
        ois=ois, video=video, adj=adj, tag1=tag1, tag2=tag2, use_case=use_case,
    )

    # 价格：高客单价，3000~25000
    price = round(random.uniform(3000, 25000), 2)

    return {
        "id": product_id,
        "name": f"{brand} {model} {adj}",
        "category": "数码相机",
        "price": price,
        "brand": brand,
        "specs": specs,
        "tags": tags,
        "description": description,
        "_meta": {"px": px, "sensor": sensor, "ois": ois,
                  "video": video, "adj": adj, "use_case": use_case},
    }


def _generate_camera_reviews(product: dict, n: int) -> list:
    meta = product["_meta"]
    reviews = []
    for i in range(n):
        tmpl = random.choice(CAMERA_REVIEW_TEMPLATES)
        text = tmpl.format(
            brand=product["brand"],
            model=product["name"].split(" ")[1] if len(product["name"].split(" ")) > 1 else "",
            feature=random.choice(["自动对焦", "高感画质", "防抖系统", "眼部追踪", "连拍速度"]),
            scenario=random.choice(["弱光室内", "夜晚街道", "阴天户外", "舞台演出"]),
            weakness=random.choice(["强光逆光", "高速运动", "极端低温", "长时间录像"]),
            tag=random.choice(product["tags"]),
            compare=random.choice(CAMERA_BRANDS),
            use_case=meta["use_case"],
        )
        reviews.append({
            "review_id": str(uuid.uuid4()),
            "product_id": product["id"],
            "type": "review",
            "text": text,
        })
    return reviews


def _generate_clothing(product_id: str) -> dict:
    brand = random.choice(CLOTHING_BRANDS)
    clothing_type = random.choice(CLOTHING_TYPES)
    adj = random.choice(CLOTHING_ADJECTIVES)
    material = random.choice(CLOTHING_MATERIALS)
    color = random.choice(CLOTHING_COLORS)
    tags = _rand_tags(CLOTHING_TAGS_POOL, 3)
    tag1, tag2 = tags[0], tags[1] if len(tags) > 1 else tags[0]
    occasion = random.choice(CLOTHING_OCCASIONS)
    season = random.choice(CLOTHING_SEASONS)

    specs = {"材质": material, "颜色": color, "版型": adj, "适用季节": season}
    desc_tmpl = random.choice(CLOTHING_DESC_TEMPLATES)
    description = desc_tmpl.format(
        brand=brand, type=clothing_type, adj=adj, material=material,
        color=color, tag1=tag1, tag2=tag2, occasion=occasion, season=season,
        fit=adj,
    )

    # 价格：非标品，99~1500
    price = round(random.uniform(99, 1500), 2)

    return {
        "id": product_id,
        "name": f"{brand} {adj}{clothing_type}",
        "category": "男装",
        "price": price,
        "brand": brand,
        "specs": specs,
        "tags": tags,
        "description": description,
        "_meta": {"type": clothing_type, "material": material, "color": color,
                  "adj": adj, "occasion": occasion, "season": season},
    }


def _generate_clothing_reviews(product: dict, n: int) -> list:
    meta = product["_meta"]
    reviews = []
    for _ in range(n):
        tmpl = random.choice(CLOTHING_REVIEW_TEMPLATES)
        text = tmpl.format(
            brand=product["brand"],
            type=meta["type"],
            material=meta["material"],
            color=meta["color"],
            adj=meta["adj"],
            tag=random.choice(product["tags"]),
            occasion=meta["occasion"],
            season=meta["season"],
            height=random.randint(165, 185),
            weight=random.randint(55, 90),
            size=random.choice(["S", "M", "L", "XL", "XXL"]),
            match=random.choice(["牛仔裤", "休闲裤", "运动裤", "西裤"]),
        )
        reviews.append({
            "review_id": str(uuid.uuid4()),
            "product_id": product["id"],
            "type": "review",
            "text": text,
        })
    return reviews


def generate_products(num_products: int) -> tuple[list, list]:
    """生成商品列表和对应的评论/文档列表。"""
    products, docs = [], []

    # 相机和男装各占一半（相机数量稍多以体现高客单价场景）
    num_cameras = num_products // 2 + num_products % 2
    num_clothing = num_products - num_cameras

    for _ in range(num_cameras):
        pid = str(uuid.uuid4())
        product = _generate_camera(pid)
        n_reviews = random.randint(
            DATA_CONFIG["reviews_per_product_min"],
            DATA_CONFIG["reviews_per_product_max"],
        )
        reviews = _generate_camera_reviews(product, n_reviews)
        # 移除临时元数据
        product.pop("_meta")
        products.append(product)
        docs.extend(reviews)

    for _ in range(num_clothing):
        pid = str(uuid.uuid4())
        product = _generate_clothing(pid)
        n_reviews = random.randint(
            DATA_CONFIG["reviews_per_product_min"],
            DATA_CONFIG["reviews_per_product_max"],
        )
        reviews = _generate_clothing_reviews(product, n_reviews)
        product.pop("_meta")
        products.append(product)
        docs.extend(reviews)

    return products, docs


def generate_users() -> list:
    """基于预设画像生成用户数据。"""
    users = []
    for i, profile in enumerate(USER_PROFILES):
        users.append({
            "user_id": str(uuid.uuid4()),
            "name": profile["name"],
            "age": profile["age"],
            "gender": profile["gender"],
            "budget_level": profile["budget_level"],
            "style_preference": profile["style_preference"],
            "category_interest": profile["category_interest"],
        })
    return users


def main(num_products: int, seed: int | None):
    if seed is not None:
        random.seed(seed)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] 生成 {num_products} 个商品数据...")
    products, docs = generate_products(num_products)

    print(f"[2/3] 生成 {len(USER_PROFILES)} 个用户数据...")
    users = generate_users()

    mock_data = {
        "products": products,
        "users": users,
        "docs": docs,
    }

    print(f"[3/3] 写入 {MOCK_DATA_FILE} ...")
    with open(MOCK_DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(mock_data, f, ensure_ascii=False, indent=2)

    print(
        f"✓ 完成！商品 {len(products)} 个，用户 {len(users)} 个，"
        f"评论/文档 {len(docs)} 条，已保存至 {MOCK_DATA_FILE}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成 Mock 电商数据")
    parser.add_argument(
        "--num-products",
        type=int,
        default=DATA_CONFIG["num_products"],
        help=f"商品总数（默认 {DATA_CONFIG['num_products']}）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DATA_CONFIG["random_seed"],
        help="随机种子（默认 42，传 -1 表示随机）",
    )
    args = parser.parse_args()
    seed = None if args.seed == -1 else args.seed
    main(num_products=args.num_products, seed=seed)
