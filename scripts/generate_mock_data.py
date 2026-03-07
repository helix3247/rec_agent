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

# ─────────────────────────── 手机素材库 ───────────────────────────
PHONE_BRANDS = ["Apple", "Samsung", "Xiaomi", "Huawei", "OPPO", "vivo", "OnePlus"]
PHONE_SERIES = ["iPhone 16", "Galaxy S25", "Xiaomi 14", "Mate 60", "Find X7", "X100 Pro", "OnePlus 12"]
PHONE_ADJECTIVES = ["旗舰", "影像旗舰", "性能旗舰", "轻薄", "长续航", "游戏向"]
PHONE_TAGS_POOL = ["高刷新率", "长续航", "快充", "影像", "轻薄", "游戏", "护眼屏"]
PHONE_SPECS_TEMPLATES = [
    {"屏幕": "{screen}", "芯片": "{chip}", "电池": "{battery}", "相机": "{camera}"},
]
PHONE_SCREEN_OPTIONS = ["6.7英寸 OLED", "6.5英寸 AMOLED", "6.8英寸 LTPO OLED"]
PHONE_CHIP_OPTIONS = ["骁龙8 Gen3", "天玑9300", "A18", "麒麟9000S"]
PHONE_BATTERY_OPTIONS = ["5000mAh", "4800mAh", "5200mAh"]
PHONE_CAMERA_OPTIONS = ["5000万三摄", "4800万双摄", "1英寸主摄", "长焦潜望式"]
PHONE_USE_CASES = ["日常通勤", "摄影爱好", "手游玩家", "商务办公", "学生党"]

PHONE_DESC_TEMPLATES = [
    "{brand} {model} 定位{adj}，配备{screen}与{chip}，{battery}续航，{camera}。"
    "适合{use_case}用户，{tag1}表现突出。",
    "{brand} {model} 主打{adj}体验，{screen}屏幕，{chip}处理器，{camera}成像，"
    "{battery}电池，兼顾{tag1}与{tag2}。",
]

PHONE_REVIEW_TEMPLATES = [
    "入手{brand} {model}后，{feature}体验很惊艳，{scenario}场景续航也稳。",
    "对比了{compare}，最终选了这款，{feature}很满足，唯一小遗憾是{weakness}。",
    "屏幕{screen}观感不错，{tag}体验明显，日常{use_case}完全够用。",
    "相机{camera}确实给力，{scenario}拍照细节清晰，质感不错。",
]

# ─────────────────────────── 运动鞋素材库 ───────────────────────────
SHOE_BRANDS = ["Nike", "Adidas", "New Balance", "ASICS", "Puma", "李宁", "安踏"]
SHOE_TYPES = ["跑鞋", "训练鞋", "篮球鞋", "休闲鞋", "板鞋", "户外鞋"]
SHOE_ADJECTIVES = ["缓震", "轻量", "稳定", "透气", "耐磨", "支撑"]
SHOE_TAGS_POOL = ["缓震", "轻便", "耐磨", "透气", "稳定", "防滑", "百搭"]
SHOE_MATERIALS = ["网布", "针织", "合成革", "织物+TPU", "麂皮"]
SHOE_COLORS = ["黑白", "灰白", "蓝黑", "米白", "全黑", "蓝灰"]
SHOE_USE_CASES = ["跑步", "健身训练", "日常通勤", "球场运动", "长途步行"]

SHOE_DESC_TEMPLATES = [
    "{brand} {adj}{type}，{material}鞋面，{color}配色，{tag1}与{tag2}兼顾，适合{use_case}。",
    "{brand} {type} 主打{adj}脚感，{material}材质，{color}外观，{tag1}表现突出。",
]

SHOE_REVIEW_TEMPLATES = [
    "{brand}这双{type}{adj}感很明显，{use_case}穿着舒服，{tag}表现不错。",
    "上脚轻盈，{material}透气，{color}很百搭，适合{use_case}。",
    "缓震不错，但在{weakness}场景稍弱，总体性价比高。",
    "对比过{compare}，最后选了这款，{tag}体验更适合我。",
]

# ─────────────────────────── 文档/FAQ 模板 ───────────────────────────
FAQ_TEMPLATES = [
    "Q: {product} 适合什么场景？A: 适合{use_case}场景，{tag1}表现突出。",
    "Q: {product} 续航/耐用性如何？A: {battery}表现稳定，日常使用足够。",
    "Q: {product} 有哪些核心卖点？A: {tag1}、{tag2}与{feature}是主要亮点。",
]

MANUAL_TEMPLATES = [
    "{product} 使用说明：首次使用建议完成基础设置，按需开启{feature}功能。",
    "{product} 保养建议：保持干燥清洁，避免极端环境影响{feature}表现。",
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

NAME_MALE_POOL = ["张明", "李浩", "王磊", "陈晨", "刘阳", "周峰", "吴凯", "郑宇", "赵凯", "宋涛"]
NAME_FEMALE_POOL = ["赵雪", "孙嘉", "林琪", "周雅", "唐琪", "许倩", "陈欣", "李萌", "方晴", "苏菲"]
STYLE_TAG_POOL = list(set(CLOTHING_TAGS_POOL + CAMERA_TAGS_POOL + PHONE_TAGS_POOL + SHOE_TAGS_POOL))
CATEGORY_INTEREST_POOL = ["数码相机", "男装", "手机", "运动鞋"]
BUDGET_LEVELS = ["low", "mid", "high"]


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
    for _ in range(n):
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


def _generate_phone(product_id: str) -> dict:
    brand = random.choice(PHONE_BRANDS)
    model = random.choice(PHONE_SERIES)
    adj = random.choice(PHONE_ADJECTIVES)
    screen = random.choice(PHONE_SCREEN_OPTIONS)
    chip = random.choice(PHONE_CHIP_OPTIONS)
    battery = random.choice(PHONE_BATTERY_OPTIONS)
    camera = random.choice(PHONE_CAMERA_OPTIONS)
    tags = _rand_tags(PHONE_TAGS_POOL, 3)
    tag1, tag2 = tags[0], tags[1] if len(tags) > 1 else tags[0]
    use_case = random.choice(PHONE_USE_CASES)

    specs = {"屏幕": screen, "芯片": chip, "电池": battery, "相机": camera}
    desc_tmpl = random.choice(PHONE_DESC_TEMPLATES)
    description = desc_tmpl.format(
        brand=brand, model=model, adj=adj, screen=screen, chip=chip,
        battery=battery, camera=camera, tag1=tag1, tag2=tag2, use_case=use_case,
    )

    price = round(random.uniform(1999, 8999), 2)

    return {
        "id": product_id,
        "name": f"{brand} {model} {adj}",
        "category": "手机",
        "price": price,
        "brand": brand,
        "specs": specs,
        "tags": tags,
        "description": description,
        "_meta": {"screen": screen, "chip": chip, "battery": battery,
                  "camera": camera, "adj": adj, "use_case": use_case},
    }


def _generate_phone_reviews(product: dict, n: int) -> list:
    meta = product["_meta"]
    reviews = []
    for _ in range(n):
        tmpl = random.choice(PHONE_REVIEW_TEMPLATES)
        text = tmpl.format(
            brand=product["brand"],
            model=product["name"].split(" ")[1] if len(product["name"].split(" ")) > 1 else "",
            feature=random.choice(["快充", "影像", "高刷屏", "性能"]),
            scenario=random.choice(["通勤路上", "夜景拍摄", "长时间游戏"]),
            weakness=random.choice(["重量", "散热", "价格偏高"]),
            compare=random.choice(PHONE_BRANDS),
            screen=meta["screen"],
            camera=meta["camera"],
            tag=random.choice(product["tags"]),
            use_case=meta["use_case"],
        )
        reviews.append({
            "review_id": str(uuid.uuid4()),
            "product_id": product["id"],
            "type": "review",
            "text": text,
        })
    return reviews


def _generate_shoe(product_id: str) -> dict:
    brand = random.choice(SHOE_BRANDS)
    shoe_type = random.choice(SHOE_TYPES)
    adj = random.choice(SHOE_ADJECTIVES)
    material = random.choice(SHOE_MATERIALS)
    color = random.choice(SHOE_COLORS)
    tags = _rand_tags(SHOE_TAGS_POOL, 3)
    tag1, tag2 = tags[0], tags[1] if len(tags) > 1 else tags[0]
    use_case = random.choice(SHOE_USE_CASES)

    specs = {"材质": material, "颜色": color, "特点": adj}
    desc_tmpl = random.choice(SHOE_DESC_TEMPLATES)
    description = desc_tmpl.format(
        brand=brand, type=shoe_type, adj=adj, material=material,
        color=color, tag1=tag1, tag2=tag2, use_case=use_case,
    )

    price = round(random.uniform(299, 1999), 2)

    return {
        "id": product_id,
        "name": f"{brand} {adj}{shoe_type}",
        "category": "运动鞋",
        "price": price,
        "brand": brand,
        "specs": specs,
        "tags": tags,
        "description": description,
        "_meta": {"type": shoe_type, "material": material, "color": color,
                  "adj": adj, "use_case": use_case},
    }


def _generate_shoe_reviews(product: dict, n: int) -> list:
    meta = product["_meta"]
    reviews = []
    for _ in range(n):
        tmpl = random.choice(SHOE_REVIEW_TEMPLATES)
        text = tmpl.format(
            brand=product["brand"],
            type=meta["type"],
            adj=meta["adj"],
            material=meta["material"],
            color=meta["color"],
            use_case=meta["use_case"],
            tag=random.choice(product["tags"]),
            weakness=random.choice(["雨天防滑", "长时间跑步"]),
            compare=random.choice(SHOE_BRANDS),
        )
        reviews.append({
            "review_id": str(uuid.uuid4()),
            "product_id": product["id"],
            "type": "review",
            "text": text,
        })
    return reviews


def _generate_faq_docs(product: dict, n: int) -> list:
    meta = product["_meta"]
    docs = []
    for _ in range(n):
        tmpl = random.choice(FAQ_TEMPLATES)
        text = tmpl.format(
            product=product["name"],
            use_case=meta.get("use_case", "日常使用"),
            tag1=random.choice(product["tags"]),
            tag2=random.choice(product["tags"]),
            feature=random.choice(list(product["specs"].keys())),
            battery=meta.get("battery", "续航"),
        )
        docs.append({
            "review_id": str(uuid.uuid4()),
            "product_id": product["id"],
            "type": "faq",
            "text": text,
        })
    return docs


def _generate_manual_docs(product: dict, n: int) -> list:
    docs = []
    for _ in range(n):
        tmpl = random.choice(MANUAL_TEMPLATES)
        text = tmpl.format(
            product=product["name"],
            feature=random.choice(list(product["specs"].keys())),
        )
        docs.append({
            "review_id": str(uuid.uuid4()),
            "product_id": product["id"],
            "type": "manual",
            "text": text,
        })
    return docs


def generate_products(num_products: int) -> tuple[list, list]:
    """生成商品列表和对应的评论/文档列表。"""
    products, docs = [], []

    category_pool = (
        ["camera"] * 30 +
        ["clothing"] * 30 +
        ["phone"] * 20 +
        ["shoe"] * 20
    )

    for _ in range(num_products):
        pid = str(uuid.uuid4())
        category = random.choice(category_pool)
        if category == "camera":
            product = _generate_camera(pid)
            review_fn = _generate_camera_reviews
        elif category == "clothing":
            product = _generate_clothing(pid)
            review_fn = _generate_clothing_reviews
        elif category == "phone":
            product = _generate_phone(pid)
            review_fn = _generate_phone_reviews
        else:
            product = _generate_shoe(pid)
            review_fn = _generate_shoe_reviews

        n_reviews = random.randint(
            DATA_CONFIG["reviews_per_product_min"],
            DATA_CONFIG["reviews_per_product_max"],
        )
        n_faq = random.randint(
            DATA_CONFIG["faq_per_product_min"],
            DATA_CONFIG["faq_per_product_max"],
        )
        n_manual = random.randint(
            DATA_CONFIG["manual_per_product_min"],
            DATA_CONFIG["manual_per_product_max"],
        )
        reviews = review_fn(product, n_reviews)
        faqs = _generate_faq_docs(product, n_faq)
        manuals = _generate_manual_docs(product, n_manual)

        product.pop("_meta")
        products.append(product)
        docs.extend(reviews + faqs + manuals)

    return products, docs


def _generate_random_user() -> dict:
    gender = random.choice(["male", "female"])
    name_pool = NAME_MALE_POOL if gender == "male" else NAME_FEMALE_POOL
    return {
        "user_id": str(uuid.uuid4()),
        "name": random.choice(name_pool),
        "age": random.randint(18, 50),
        "gender": gender,
        "budget_level": random.choice(BUDGET_LEVELS),
        "style_preference": _rand_tags(STYLE_TAG_POOL, random.randint(1, 3)),
        "category_interest": _rand_tags(CATEGORY_INTEREST_POOL, random.randint(1, 3)),
    }


def generate_users(num_users: int) -> list:
    """基于预设画像 + 随机扩展生成用户数据。"""
    users = []
    for profile in USER_PROFILES:
        users.append({
            "user_id": str(uuid.uuid4()),
            "name": profile["name"],
            "age": profile["age"],
            "gender": profile["gender"],
            "budget_level": profile["budget_level"],
            "style_preference": profile["style_preference"],
            "category_interest": profile["category_interest"],
        })
    while len(users) < num_users:
        users.append(_generate_random_user())
    return users


def generate_interactions(users: list, products: list) -> list:
    interactions = []
    action_pool = ["view", "like", "cart", "purchase"]
    action_weights = [0.55, 0.2, 0.15, 0.1]
    for user in users:
        n = random.randint(
            DATA_CONFIG["interactions_per_user_min"],
            DATA_CONFIG["interactions_per_user_max"],
        )
        for _ in range(n):
            product = random.choice(products)
            action = random.choices(action_pool, weights=action_weights, k=1)[0]
            interactions.append({
                "user_id": user["user_id"],
                "product_id": product["id"],
                "action": action,
                "session_id": str(uuid.uuid4()),
            })
    return interactions


def generate_orders(users: list, products: list) -> list:
    orders = []
    status_pool = ["pending", "paid", "shipped", "done", "cancelled"]
    for user in users:
        n = random.randint(
            DATA_CONFIG["orders_per_user_min"],
            DATA_CONFIG["orders_per_user_max"],
        )
        for _ in range(n):
            product = random.choice(products)
            quantity = random.randint(1, 3)
            total_price = round(product["price"] * quantity, 2)
            orders.append({
                "order_id": str(uuid.uuid4()),
                "user_id": user["user_id"],
                "product_id": product["id"],
                "quantity": quantity,
                "total_price": total_price,
                "status": random.choice(status_pool),
            })
    return orders


def main(num_products: int, seed: int | None):
    if seed is not None:
        random.seed(seed)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] 生成 {num_products} 个商品数据...")
    products, docs = generate_products(num_products)

    print(f"[2/4] 生成 {DATA_CONFIG['num_users']} 个用户数据...")
    users = generate_users(DATA_CONFIG["num_users"])

    print("[3/4] 生成用户交互与订单数据...")
    interactions = generate_interactions(users, products)
    orders = generate_orders(users, products)

    mock_data = {
        "products": products,
        "users": users,
        "interactions": interactions,
        "orders": orders,
        "docs": docs,
    }

    print(f"[4/4] 写入 {MOCK_DATA_FILE} ...")
    with open(MOCK_DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(mock_data, f, ensure_ascii=False, indent=2)

    print(
        f"✓ 完成！商品 {len(products)} 个，用户 {len(users)} 个，"
        f"交互 {len(interactions)} 条，订单 {len(orders)} 条，"
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
    seed_value = None if args.seed == -1 else args.seed
    main(num_products=args.num_products, seed=seed_value)
