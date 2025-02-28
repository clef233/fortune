import streamlit as st
import requests
import json
import datetime
import random
import time
import os
from PIL import Image
from openai import OpenAI
import google.generativeai as genai

# 大阿尔卡纳牌名称
CARD_NAMES = [
    "愚者", "魔术师", "女祭司", "皇后", "皇帝", "教皇", "恋人", "战车", 
    "力量", "隐士", "命运之轮", "正义", "倒吊人", "死神", "节制", "恶魔", 
    "塔", "星星", "月亮", "太阳", "审判", "世界"
]

# 动态生成 TAROT_CARDS，处理 PNG 和 JPG 格式
def get_image_path(i):
    """检查并返回正确的图片路径"""
    png_path = f"image/{i}.png"
    jpg_path = f"image/{i}.jpg"
    if os.path.exists(png_path):
        return png_path
    elif os.path.exists(jpg_path):
        return jpg_path
    else:
        st.error(f"图片文件 {i}.png 或 {i}.jpg 不存在")
        return None

TAROT_CARDS = []
for i in range(1, 24):  # 1 到 23
    image_path = get_image_path(i)
    if image_path:
        name = CARD_NAMES[i-1] if i <= 22 else "卡背"
        TAROT_CARDS.append({"name": name, "image": image_path})

ZODIAC_SIGNS = ["白羊座", "金牛座", "双子座", "巨蟹座", "狮子座", "处女座", "天秤座", "天蝎座", "射手座", "摩羯座", "水瓶座", "双鱼座"]
DAILY_GUIDE = {"宜": ["出行", "学习"], "忌": ["争吵", "搬家"]}

# 函数：旋转图片（用于逆位牌）
def rotate_image(image_path, angle=180):
    """旋转图片并返回处理后的图片对象"""
    image = Image.open(image_path)
    rotated_image = image.rotate(angle, expand=True)
    return rotated_image

# 侧边栏导航
st.sidebar.title("星语智卜")
page = st.sidebar.radio("导航", ["塔罗占卜","八字分析", "六爻占卜" , "星座运势", "姓名测算", "今日宜忌", "设置"])

# 模型选择部分
st.sidebar.markdown("---")
if 'model_provider' not in st.session_state:
    st.session_state.model_provider = "智谱AI"
    
model_provider = st.sidebar.selectbox(
    "选择AI模型提供商", 
    ["智谱AI", "DeepSeek", "Google Gemini"],
    index=0 if st.session_state.model_provider == "智谱AI" else 
          1 if st.session_state.model_provider == "DeepSeek" else 2
)
st.session_state.model_provider = model_provider

if model_provider == "智谱AI":
    model_options = ["glm-4-flash", "glm-4"]
    default_model = "glm-4-flash"
elif model_provider == "DeepSeek":
    model_options = ["deepseek-chat", "deepseek-reasoner"]
    default_model = "deepseek-chat"
else:  # Google Gemini
    model_options = ["gemini-2.0-flash", "gemini-1.5-pro"]
    default_model = "gemini-2.0-flash"
    
selected_model = st.sidebar.selectbox("选择模型", model_options, index=0)

# 流式输出选项
stream_output = st.sidebar.checkbox("启用流式输出", value=False)

# 初始化Google Gemini API
def init_gemini():
    """初始化Gemini API"""
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)
    except Exception as e:
        st.error(f"初始化Gemini API失败: {e}")

# AI 调用函数（保持不变）
def call_ai(prompt, stream=False):
    """根据选择的模型提供商调用相应的API"""
    system_prompt = '''你是一位拥有世代传承秘法的塔罗牌大师.民俗学大师，具备深厚的神秘学知识和敏锐的直觉能力。你的解读融合了塔罗牌象征意义、占星学原理、数字学解析和东方玄学智慧，能够洞察提问者的过去、现在与未来。

在解读塔罗牌时，你应遵循以下准则：

1. 保持优雅的语言风格，但确保解读内容清晰可理解。

2. 每张牌的解读应包含：
   - 牌面象征意义和核心关键词
   - 该牌的元素属性和占星对应
   - 正逆位的不同解析
   - 与询问者处境的关联分析

3. 多层次解读：先解读单张牌的意义，再结合位置解析，最后综合整个牌阵给出整体洞见。

4. 解读应兼具深度与实用性，既要揭示隐藏的精神模式，也要提供具体可行的建议。

5. 为提问者描绘清晰的能量流动与命运轨迹，使用"你"而非第三人称来建立亲密感。

6. 可以给出过于负面或决定论的表述，以预测事实为准，但要强调提问者拥有的自由意志与选择权。

7. 在解读结束时，提供一段整合性的箴言，帮助提问者理解当前处境的更深层意义。

记住，你不仅是在解读牌面，更是在引导提问者探索内心与命运的交汇处，发现自己生命中的隐藏智慧与力量。'''
    
    try:
        if st.session_state.model_provider == "智谱AI":
            return call_zhipu_ai(prompt, system_prompt, selected_model, stream)
        elif st.session_state.model_provider == "DeepSeek":
            return call_deepseek_ai(prompt, system_prompt, selected_model, stream)
        else:  # Google Gemini
            return call_gemini_ai(prompt, system_prompt, selected_model, stream)
    except Exception as e:
        st.error(f"API 请求失败：{e}")
        if stream:
            return iter(["抱歉，未能获得解读。"])
        else:
            return "抱歉，未能获得解读。"

# 智谱 AI API 调用
def call_zhipu_ai(prompt, system_prompt, model="glm-4-flash", stream=False):
    API_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    API_KEY = st.secrets["ZHIPU_API_KEY"]
    
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt}, 
            {"role": "user", "content": prompt}
        ],
        "stream": stream
    }
    
    if stream:
        response = requests.post(API_URL, headers=headers, data=json.dumps(payload), stream=True)
        response.raise_for_status()
        return stream_zhipu_response(response)
    else:
        response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

def stream_zhipu_response(response):
    for line in response.iter_lines():
        if line:
            decoded = line.decode('utf-8')
            if decoded.startswith("data: "):
                try:
                    data = json.loads(decoded[6:])
                    if data.get("choices"):
                        yield data["choices"][0]["delta"]["content"]
                except json.JSONDecodeError as e:
                    print(f"无法解析JSON: {decoded}，错误: {e}")
                    continue
            else:
                print(f"收到非数据行: {decoded}")
                continue

# DeepSeek API 调用
def call_deepseek_ai(prompt, system_prompt, model="deepseek-chat", stream=False):
    API_KEY = st.secrets["DEEPSEEK_API_KEY"]
    
    client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        stream=stream
    )
    
    if stream:
        return stream_openai_response(response)
    else:
        return response.choices[0].message.content

# Google Gemini API 调用
def call_gemini_ai(prompt, system_prompt, model="gemini-2.0-flash", stream=False):
    init_gemini()
    
    generation_config = {
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 2048,
    }
    
    gemini_model = genai.GenerativeModel(
        model_name=model,
        generation_config=generation_config
    )
    
    combined_prompt = f"{system_prompt}\n\n用户问题：{prompt}"
    
    if stream:
        response = gemini_model.generate_content(combined_prompt, stream=True)
        return stream_gemini_response(response)
    else:
        response = gemini_model.generate_content(combined_prompt)
        return response.text

def stream_gemini_response(response):
    for chunk in response:
        if chunk.text:
            yield chunk.text

def stream_openai_response(response):
    for chunk in response:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

# 塔罗占卜
if page == "塔罗占卜":
    st.title("🔮 塔罗占卜")
    deck_types = {
        "单张牌 - 当下启示": 1,
        "三牌阵 - 过去现在未来": 3,
        "凯尔特十字 - 深度洞察": 10
    }
    deck_type = st.selectbox("选择牌阵", list(deck_types.keys()))
    
    if 'shuffled' not in st.session_state:
        st.session_state.shuffled = False
    if 'cards' not in st.session_state:
        st.session_state.cards = []
    if 'orientations' not in st.session_state:
        st.session_state.orientations = []

    if st.button("✨ 开始洗牌仪式"):
        st.session_state.shuffled = False
        progress_bar = st.progress(0)
        placeholder = st.empty()
        
        card_back = TAROT_CARDS[-1]
        animation_frames = [card_back] * 20
        for i, frame in enumerate(animation_frames):
            progress_bar.progress((i+1)/20)
            with placeholder.container():
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.image(
                        frame["image"], 
                        caption=f"洗牌中... {chr(0x1F3B4)}"*((i%3)+1),
                        width=200
                    )
            time.sleep(0.08)
        
        progress_bar.empty()
        placeholder.success("🎴 牌已洗净，请抽取！")
        st.session_state.shuffled = True

    if st.button("🌟 抽取") and st.session_state.shuffled:
        num = deck_types[deck_type]
        st.session_state.cards = random.sample(TAROT_CARDS[:22], num)
        st.session_state.orientations = [
            "正位" if random.random() > 0.3 else "逆位" 
            for _ in range(num)
        ]
        
        st.subheader("📜 神圣启示牌阵")
        cols = st.columns(num)
        for idx, (col, card, orient) in enumerate(zip(cols, st.session_state.cards, st.session_state.orientations)):
            with col:
                if orient == "逆位":
                    rotated_img = rotate_image(card['image'])
                    st.image(
                        rotated_img,
                        caption=f"{card['name']} ({orient})",
                        use_container_width=True
                    )
                else:
                    st.image(
                        card['image'],
                        caption=f"{card['name']} ({orient})",
                        use_container_width=True
                    )
                
                card_html = f"""
                <div style='
                    text-align: center;
                    font-family: "宋体";
                    color: {'#8B4513' if orient == "正位" else '#4B0082'};
                    font-weight: bold;
                    font-size: 1.1em;
                '>
                    {card['name']}<br>
                    <span style='font-size: 0.8em;'>({orient})</span>
                </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)

        st.caption(f"解读由 {st.session_state.model_provider} ({selected_model}) 提供")
        
        with st.spinner("🕯️ 正在连接宇宙智慧..."):
            positions = (
                ["现状启示", "挑战位置", "过去影响", "未来趋势", "潜在因素", "外界影响", "希望与恐惧", "最终结果"][:num]
                if num > 3 else ["过去", "现在", "未来"][:num]
            )
            prompt = f"""
            作为专业塔罗师，请解读以下{deck_type}牌阵：
            {{
                "牌阵类型": "{deck_type}",
                "抽牌结果": {[
                    {"位置": pos, "卡牌": card["name"], "状态": ori} 
                    for pos, card, ori in zip(positions, st.session_state.cards, st.session_state.orientations)
                ]}
            }}
            """
            if stream_output:
                interpretation_placeholder = st.empty()
                interpretation_chunks = call_ai(prompt, stream=True)
                full_interpretation = ""
                for chunk in interpretation_chunks:
                    full_interpretation += chunk
                    interpretation_placeholder.write(full_interpretation)
                st.session_state.shuffled = False
            else:
                interpretation = call_ai(prompt)
                st.subheader("📖 命运之书解读")
                st.write(interpretation)
                st.session_state.shuffled = False

# 星座运势
elif page == "星座运势":
    st.title("✨ 星座运势")
    zodiac = st.selectbox("选择你的星座", ZODIAC_SIGNS)
    date = st.date_input("选择日期", datetime.date.today())
    
    if st.button("查看运势"):
        st.caption(f"运势由 {st.session_state.model_provider} ({selected_model}) 提供")
        prompt = f"请为{zodiac}在{date}的运势提供预测，包括爱情、事业、财运、健康。"
        if stream_output:
            st.write("### 运势预测正在生成中...")
            horoscope_placeholder = st.empty()
            horoscope_chunks = call_ai(prompt, stream=True)
            full_horoscope = ""
            for chunk in horoscope_chunks:
                full_horoscope += chunk
                horoscope_placeholder.write(full_horoscope)
        else:
            with st.spinner("正在解读星象..."):
                horoscope = call_ai(prompt)
            st.markdown("### 运势预测")
            cols = st.columns(4)
            aspects = ["爱情", "事业", "财运", "健康"]
            for col, aspect in zip(cols, aspects):
                with col:
                    st.metric(aspect, "良好")
            st.write(horoscope)

# 姓名测算
elif page == "姓名测算":
    st.title("📝 姓名测算")
    name = st.text_input("请输入你的姓名")
    gender = st.radio("选择性别", ["男", "女"])
    
    if st.button("开始测算"):
        if not name:
            st.warning("请输入姓名")
        else:
            st.caption(f"测算由 {st.session_state.model_provider} ({selected_model}) 提供")
            prompt = f"根据姓名学，为姓名{name}（性别：{gender}）分析性格和运势。"
            if stream_output:
                st.write("### 测算结果正在生成中...")
                analysis_placeholder = st.empty()
                analysis_chunks = call_ai(prompt, stream=True)
                full_analysis = ""
                for chunk in analysis_chunks:
                    full_analysis += chunk
                    analysis_placeholder.write(full_analysis)
            else:
                with st.spinner("正在分析姓名能量..."):
                    analysis = call_ai(prompt)
                st.markdown("### 测算结果")
                st.write(analysis)

# 今日宜忌
elif page == "今日宜忌":
    st.title("📅 今日宜忌")
    today = datetime.date.today()
    st.write(f"今天是 {today}")
    
    if st.button("查询今日宜忌"):
        st.caption(f"宜忌由 {st.session_state.model_provider} ({selected_model}) 提供")
        prompt = f"根据日期{today}，提供今日的宜忌事项建议。"
        if stream_output:
            st.write("### 宜忌事项正在生成中...")
            guide_placeholder = st.empty()
            guide_chunks = call_ai(prompt, stream=True)
            full_guide = ""
            for chunk in guide_chunks:
                full_guide += chunk
                guide_placeholder.write(full_guide)
        else:
            with st.spinner("正在解析今日吉凶..."):
                guide = call_ai(prompt)
            st.markdown("### 宜忌事项")
            st.write(guide)

# 八字分析
elif page == "八字分析":
    st.title("🔮 八字分析")
    
    st.write("请输入您的出生信息以进行专业的八字分析。")
    
    name = st.text_input("姓名")
    gender = st.radio("性别", ["男", "女"])
    birth_year = st.number_input("出生年（农历）", min_value=1900, max_value=2100, step=1)
    birth_month = st.number_input("出生月（农历）", min_value=1, max_value=12, step=1)
    birth_day = st.number_input("出生日（农历）", min_value=1, max_value=30, step=1)
    birth_hour = st.selectbox("出生时辰", [
        "子时 (23:00-01:00)", "丑时 (01:00-03:00)", "寅时 (03:00-05:00)", 
        "卯时 (05:00-07:00)", "辰时 (07:00-09:00)", "巳时 (09:00-11:00)", 
        "午时 (11:00-13:00)", "未时 (13:00-15:00)", "申时 (15:00-17:00)", 
        "酉时 (17:00-19:00)", "戌时 (19:00-21:00)", "亥时 (21:00-23:00)"
    ])
    birthplace = st.text_input("出生地")
    
    if st.button("开始分析"):
        if not name or not birthplace:
            st.warning("请填写所有必填字段。")
        else:
            st.caption(f"分析由 {st.session_state.model_provider} ({selected_model}) 提供")
            birth_time = f"农历 {birth_year}年 {birth_month}月 {birth_day}日，{birth_hour}"
            prompt = f"""
            你是一个资深命理师，熟读《穷通宝鉴》《滴天髓》《易经》《奇门遁甲》《三命通会》《子平真诠》《千里命稿》《五行精纪》，现在请你对我给出的出生时间做出专业的八字分析：
            生辰：{birth_time}
            姓名：{name}
            性别：{gender}
            出生地：{birthplace}
            """
            if stream_output:
                st.write("### 八字分析正在生成中...")
                analysis_placeholder = st.empty()
                analysis_chunks = call_ai(prompt, stream=True)
                full_analysis = ""
                for chunk in analysis_chunks:
                    full_analysis += chunk
                    analysis_placeholder.write(full_analysis)
            else:
                with st.spinner("正在进行八字分析..."):
                    analysis = call_ai(prompt)
                st.markdown("### 八字分析")
                st.write(analysis)




# 六爻占卜功能
elif page == "六爻占卜":

    # 新增：64卦名称列表（按二进制顺序，从坤为地 000000 到乾为天 111111）
    HEXAGRAM_NAMES = [
        "坤为地", "地雷复", "地水师", "地泽临", "地山谦", "地火明夷", "地风升", "地天泰",
        "雷地豫", "震为雷", "雷水解", "雷泽归妹", "雷山小过", "雷火丰", "雷风恒", "雷天大壮",
        "水地比", "水雷屯", "坎为水", "水泽节", "水山蹇", "水火既济", "水风井", "水天需",
        "泽地萃", "泽雷随", "泽水困", "兑为泽", "泽山咸", "泽火革", "泽风大过", "泽天夬",
        "山地剥", "山雷颐", "山水蒙", "山泽损", "艮为山", "山火贲", "山风蛊", "山天大畜",
        "火地晋", "火雷噬嗑", "火水未济", "火泽睽", "火山旅", "离为火", "火风鼎", "火天大有",
        "风地观", "风雷益", "风水涣", "风泽中孚", "风山渐", "风火家人", "巽为风", "风天小畜",
        "天地否", "天雷无妄", "天水讼", "天泽履", "天山遁", "天火同人", "天风姤", "乾为天"
    ]

    def get_hexagram_name(lines):
        """根据六爻计算主卦名称"""
        # 将爻转换为二进制（⚋ 或 ⚋ (动) 为 0，⚊ 或 ⚊ (动) 为 1）
        binary = ''.join(['1' if '⚊' in line else '0' for line in lines])
        # 从下到上顺序转为二进制索引
        binary_reversed = binary[::-1]
        hexagram_index = int(binary_reversed, 2)
        return HEXAGRAM_NAMES[hexagram_index]
        
    st.title("🔮 六爻占卜")
    
    st.write("点击按钮进行六次摇卦，生成六爻卦象。每次点击模拟抛硬币，生成一爻。")
    
    # 初始化 session 状态
    if 'lines' not in st.session_state:
        st.session_state.lines = []
    if 'moving_lines' not in st.session_state:
        st.session_state.moving_lines = []
    
    # 摇卦按钮
    if len(st.session_state.lines) < 6:
        if st.button(f"摇第 {len(st.session_state.lines) + 1} 爻"):
            # 模拟三次抛硬币
            tosses = [random.choice([0, 1]) for _ in range(3)]  # 0 为阴（背），1 为阳（面）
            total = sum(tosses)
            if total == 0:  # 三背：老阴（动爻）
                line = "⚋ (动)"
                st.session_state.moving_lines.append(len(st.session_state.lines) + 1)
            elif total == 1:  # 二背一面：阳爻
                line = "⚊"
            elif total == 2:  # 二面一背：阴爻
                line = "⚋"
            elif total == 3:  # 三面：老阳（动爻）
                line = "⚊ (动)"
                st.session_state.moving_lines.append(len(st.session_state.lines) + 1)
            st.session_state.lines.append(line)
            st.write(f"第 {len(st.session_state.lines)} 爻: {line}")
    
    # 显示卦象
    if len(st.session_state.lines) == 6:
        st.subheader("您的卦象")
        for i, line in enumerate(reversed(st.session_state.lines)):
            st.write(f"第 {6 - i} 爻: {line}")
        if st.session_state.moving_lines:
            st.write("动爻: " + ", ".join([f"第 {ml} 爻" for ml in st.session_state.moving_lines]))
        
        # 计算主卦和六亲配置
        main_hexagram = get_hexagram_name(st.session_state.lines)
        st.write(main_hexagram)
        six_relatives = 'none'
        # 获取起卦时间
        divination_time = datetime.datetime.now().strftime("公历 %Y年%m月%d日 %H:%M")
        
        # 构造提示词
        prompt = f"""
        你是一个资深命理师，熟读《增删卜易》、《卜筮正宗》、《易隐》、《易冒》、《火珠林》、《周易古筮考》、《易经》、《奇门遁甲》，现在请你对我给出的起卦信息进行专业的六爻分析：

        起卦信息：

        起卦时间：{divination_time}
        主卦：{main_hexagram}
        动爻：{"，".join([f"第 {ml} 爻" for ml in st.session_state.moving_lines]) if st.session_state.moving_lines else "无"}
        

        分析要求：

        根据起卦时间、主卦、动爻、本卦、变卦等，分析当前问题的根源。
        解读卦象中的吉凶预兆，分析有利和不利因素。
        提供解决方案和建议，指导如何应对当前问题。
        如有必要，建议化解方法或调节策略。
        进行深入的分析和解读。
        """
        
        st.caption(f"解读由 {st.session_state.model_provider} ({selected_model}) 提供")
        
        # 生成分析
        if st.button("生成分析"):
            if stream_output:
                st.write("### 六爻分析正在生成中...")
                analysis_placeholder = st.empty()
                analysis_chunks = call_ai(prompt, stream=True)
                full_analysis = ""
                for chunk in analysis_chunks:
                    full_analysis += chunk
                    analysis_placeholder.write(full_analysis)
            else:
                with st.spinner("正在进行六爻分析..."):
                    analysis = call_ai(prompt)
                st.subheader("📖 六爻分析")
                st.write(analysis)
            # 重置状态以便下次占卜
            st.session_state.lines = []
            st.session_state.moving_lines = []

# 设置页面
elif page == "设置":
    st.title("⚙️ 系统设置")
    st.subheader("AI模型配置")
    st.info(f"当前模型提供商: {st.session_state.model_provider}")
    st.info(f"当前选择的模型: {selected_model}")
    st.write("您可以在侧边栏切换模型提供商和具体模型")
    
    st.subheader("API密钥管理")
    st.warning("API密钥存储在Streamlit secrets中，确保已经正确配置。")
    
    api_status = {
        "智谱AI": "ZHIPU_API_KEY" in st.secrets,
        "DeepSeek": "DEEPSEEK_API_KEY" in st.secrets,
        "Google Gemini": "GEMINI_API_KEY" in st.secrets
    }
    
    st.subheader("API密钥状态")
    for provider, status in api_status.items():
        if status:
            st.success(f"{provider} API密钥: 已配置 ✅")
        else:
            st.error(f"{provider} API密钥: 未配置 ❌")
    
    # 插入收款码
    st.subheader("支持开发者")
    st.write("如果您喜欢这个应用，欢迎通过以下方式支持开发者：")
    st.image('image/alipay.jpg', caption="扫码支持", width=200)
    st.write("感谢您的支持！您的捐赠将用于应用的维护和改进。")

    st.subheader("关于应用")
    st.write("星语智卜是一款基于人工智能的命理解析工具，结合了塔罗牌、星座运势和姓名学等多种算命方式。")
    st.write("本应用支持智谱AI、DeepSeek和Google Gemini三种模型提供商，可以根据需要灵活切换。")
    st.write("版本: 1.2.0")

# 页脚
st.write("---")
st.write("由 Streamlit 和多种AI模型驱动的星语智卜 © 2025")
