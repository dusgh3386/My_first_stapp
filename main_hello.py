import streamlit as st

st.set_page_config(
    page_title="K팝 데몬 헌터스 팬덤 분석",
    layout="wide"
)

st.title("K팝 데몬 헌터스 팬덤 형성 요인 분석")
st.write("학번: C321017    이름: 김연호")

st.divider()

st.sidebar.header("🔍 분석 옵션")

date_range = st.sidebar.date_input(
    "분석 기간 선택",
    []
)

top_n = st.sidebar.slider(
    "키워드 개수 선택",
    min_value=10,
    max_value=50,
    value=30,
    step=5
)

show_global = st.sidebar.checkbox(
    "글로벌 성과 키워드 포함",
    value=True
)

wc_max_words = st.sidebar.slider(
    "워드클라우드 최대 단어 수",
    min_value=50,
    max_value=300,
    value=150,
    step=10
)

min_edge = st.sidebar.slider(
    "네트워크 최소 연결 빈도",
    min_value=1,
    max_value=10,
    value=3
)

st.header("1️⃣ Seaborn을 이용한 시점별 기사 수 추이 분석")
st.write(
    "케이팝 데몬 헌터스 관련 온라인 기사 데이터를 기반으로 이슈가 어떤 시점부터 집중적으로 확산되었는지를 확인하는 Seaborn 그래프. (koreanized-matplotlib가 설치되지 않아 영어로 제목 등을 작성하였습니다. seaborn 그래프 코드에서 날짜 범위 필터링 코드는 AI를 사용하였습니다.)"
)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("F:/Users/김 연호/Desktop/학교/2025-2/데이터시각화/data_데이터시각화/케데헌.csv")
df["pubDate"] = pd.to_datetime(df["pubDate"]).dt.date

if len(date_range) == 2:
    df = df[
        (df["pubDate"] >= date_range[0]) &
        (df["pubDate"] <= date_range[1])
    ]

date_count = (
    df.groupby("pubDate")
    .size()
    .reset_index(name="count")
)

fig, ax = plt.subplots(figsize=(10, 4))
sns.lineplot(data=date_count, x="pubDate", y="count", marker="o", ax=ax)
ax.set_xlabel("date")
ax.set_ylabel("number of articles")
ax.set_title("Trend of K-Pop Demon Hunters Article Counts Over Time")

st.pyplot(fig)

st.write(
    "위의 Seaborn 그래프는 케이팝 데몬 헌터스와 관련된 기사 수가 날짜별로 어떻게 변화했는지를 보여준다. 11월 말부터 기사 수가 점차 증가하며 작품에 대한 관심이 커지고 있음을 확인할 수 있다. 중간에 기사 수가 일시적으로 감소하는 구간도 나타나지만, 전체적으로 보면 12월 중순까지는 증감을 반복하면서도 증가하는 추세를 보인다. 이를 통해 케이팝 데몬 헌터스에 대한 관심이 점점 커졌음을 알 수 있다."
)


st.divider()

st.header("2️⃣ WordCloud를 이용한 단어 빈도 시각화 분석")
st.write(
    "케이팝 데몬 헌터스 관련 온라인 기사 데이터를 기반으로 기사 제목과 본문에서 자주 등장한 키워드를 시각화한 WordCloud 그래프. '케이팝 데몬 헌터스', '케데헌' 등 고유명사와 불용어 제거 및 텍스트 전처리를 통해 의미 있는 단어만을 추출하였으며, 한글 표현을 위해 WordCloud에 한글 폰트, 나눔고딕 폰트를 적용하였다. WordCloud 생성 및 전처리 과정은 강의록의 텍스트 시각화 코드 흐름을 기반으로 구현하였다.(텍스트 정화, 불용어들은 AI로 작성하고 분석 목적에 맞게 편집하였습니다.)"
)

import pandas as pd
import re
from wordcloud import WordCloud, STOPWORDS

df["title"] = df["title"].fillna("").astype(str)
df["description"] = df["description"].fillna("").astype(str)
df["text"] = (df["title"] + " " + df["description"]).str.strip()

text_kdh = " ".join(df["text"].tolist())

text_kdh_clean = text_kdh
text_kdh_clean = re.sub(r"&quot;|quot", " ", text_kdh_clean)
text_kdh_clean = re.sub(r"&lt;|lt", " ", text_kdh_clean)
text_kdh_clean = re.sub(r"&gt;|gt", " ", text_kdh_clean)
text_kdh_clean = re.sub(r"&amp;|amp", " ", text_kdh_clean)
text_kdh_clean = re.sub(r"[\'\"“”‘’]", " ", text_kdh_clean)
text_kdh_clean = re.sub(r"[^0-9A-Za-z가-힣\s]", " ", text_kdh_clean)
text_kdh_clean = re.sub(r"\s+", " ", text_kdh_clean).strip()

stop_words_kdh = [
    # 1. 주제어 및 고유명사 (분석 목적에 따라 제거)
    '데몬', '헌터스', '케이팝', 'k', 'K', 'K팝', '케데헌', '팝', '애니메이션', '넷플릭스', 
    '영화', 'k팝', 'ost', 'demon', 'hunters', '걸그룹', '아이돌', '소다', 
    '골든', '골든글로브', '빌보드', '박찬욱', '강', '매기',
    
    # 2. 조사, 접속사 및 어미 (한국어 일반 불용어)
    '의', '가', '이', '와', '과', '에', '를', '을', '는', '은', '한', '있는', '없다',
    
    # 3. 노이즈 및 일반 빈출 단어
    'quot', '등', '위', '일', '전', '년', '개', '최고', '특히', '컬처', '부문'
]

STOPWORDS.update(stop_words_kdh)

words_list = text_kdh_clean.split()
words_list = [w for w in words_list if w not in STOPWORDS]
text_kdh_clean2 = " ".join(words_list)

#이미지파일을 불러와 ndarray로 변환
import numpy as np
from PIL import Image

#마스크가 될 이미지 불러오기
image = Image.open(
    "F:/Users/김 연호/Desktop/학교/2025-2/데이터시각화/data_데이터시각화/cross_new.png" # 이미지 파일 경로
    ).resize(size=(800, 800)) #이미지 크기 지정
wc_mask=Image.new("RGB", image.size, (255,255,255))
wc_mask.paste(im=image, mask=image)
wc_mask = np.array(wc_mask)

han_font_path = "F:/Users/김 연호/Downloads/nanum-all_new/나눔 글꼴/나눔고딕/NanumFontSetup_OTF_GOTHIC/NanumGothic.otf"

def showWordCloudBasic(wc):
    fig = plt.figure(figsize=(8, 5))
    plt.imshow(wc)
    plt.axis("off")
    return fig

words_kdh = WordCloud(
    font_path=han_font_path,
    max_words=wc_max_words,
    stopwords=STOPWORDS,
    background_color="black",
    mask=wc_mask,
    colormap="coolwarm"
).generate(text_kdh_clean2)

fig = showWordCloudBasic(words_kdh)
st.pyplot(fig)

st.write(
    "위의 WordCloud 시각화 결과를 보면 가장 크게 '글로벌'이라는 단어를 확인할 수 있다. 이는 케이팝데몬헌터스가 국내 뿐 아니라 해외에서도 엄청난 열풍이라는 것과 글로벌 이슈라는 것을 의미한다. 또 Golden, Soda Pop 등 케이팝 데몬 헌터스의 노래 제목들도 매우 큰 글씨인 것을 보아 케데헌의 인기에 노래가 중요한 역할을 했다는 것을 알 수 있다. 따라서 케이팝 데몬 헌터스는 작품 자체의 재미뿐 아니라 음악적인 측면에서도 높은 완성도와 대중성을 갖추었다는 것을 의미한다."
)


st.divider()

st.header("3️⃣ Altair를 이용한 키워드 빈도 분석")
st.write(
    "앞서 WordCloud 에서 전처리 하였던 기사 데이터에서 키워드 빈도를 집계하여 키워드의 빈도를 Altair 막대그래프로 시각화하였다. 그래프의 가독성을 위해 가로 막대 그래프로 시각화하였고,  빈도순으로 정렬하여 한눈에 확인할 수 있도록 하였다. (altair 그래프의 오류를 AI의 도움을 받아 해결하였습니다.)"
)
import pandas as pd
import altair as alt

kw_df = (
    pd.Series(text_kdh_clean2.split())
    .value_counts()
    .head(top_n)
    .reset_index()
)
kw_df.columns = ["keyword", "count"]

c = (
    alt.Chart(kw_df)
    .mark_bar()
    .encode(
        x="count",
        y=alt.Y("keyword", sort="-x"),
        tooltip=["keyword", "count"]
    )
)

st.altair_chart(c, use_container_width=True)

st.write(
    "위의 Altair 막대그래프를 통해 케이팝 데몬 헌터스 관련 온라인 기사에서 나오는 키워드들의 빈도를 확인할 수 있다. 앞서 WordCloud의 결과 해석과 마찬가지로, 글로벌이 가장 큰 빈도를 보였다. WordCloud의 시각화와 다른 점은 키워드들간의 비교가 훨씬 더 쉽다는 것이다. WordCloud는 키워드의 빈도수나 키워드 간에 무엇이 더 큰지 알기가 쉽지 않지만, Altair 그래프는 막대로 대략의 수치와 함께 쉽게 알 수 있다.  그래프로 케이팝 데몬 헌터스는 작품성과 음악성을 모두 갖추었다는 것을 다시 한 번 확인할 수 있었다."
)   

st.divider()

st.header("4️⃣ Networkx를 이용한 키워드 네트워크 구조 분석")
st.write(
    "키워드 간 동시 출현 관계를 Networkx를 이용하여 네트워크 구조로 시각화하였다. 각 키워드는 노드를 의미하고 키워드 쌍은 엣지로 이어진다.(koreanized-matplotlib가 안 되어 AI를 사용하였으나 한글로 설정하지 못했습니다.)"
)
import re
import itertools
from collections import Counter
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

top_n_net = st.sidebar.slider("네트워크 Top N 키워드", 10, 60, 30)
min_edge = st.sidebar.slider("최소 동시출현(엣지 가중치) 기준", 1, 10, 2)

def clean_tokens(s):
    s = str(s)
    s = re.sub(r"&quot;|quot", " ", s)
    s = re.sub(r"&lt;|lt", " ", s)
    s = re.sub(r"&gt;|gt", " ", s)
    s = re.sub(r"&amp;|amp", " ", s)
    s = re.sub(r"[\'\"“”‘’]", " ", s)
    s = re.sub(r"[^0-9A-Za-z가-힣\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    words = [w for w in s.split() if w and w not in STOPWORDS and len(w) >= 2]
    return words

docs_tokens = df["text"].fillna("").astype(str).apply(clean_tokens)

all_words = list(itertools.chain.from_iterable(docs_tokens.tolist()))
top_words = (
    pd.Series(all_words)
    .value_counts()
    .head(top_n_net)
    .index
    .tolist()
)

edges = Counter()

for tokens in docs_tokens:
    tokens = [t for t in tokens if t in top_words]
    tokens = list(dict.fromkeys(tokens))
    for node1, node2 in itertools.combinations(sorted(tokens), 2):
        edges[(node1, node2)] += 1

filtered_edges = {k: v for k, v in edges.items() if v >= min_edge}

G = nx.Graph()

weighted_edges = [
    (node1, node2, weight)
    for (node1, node2), weight in filtered_edges.items()
]
G.add_weighted_edges_from(weighted_edges)

if G.number_of_nodes() != 0:
    pos_spring = nx.spring_layout(
        G,
        k=0.3,
        iterations=50,
        seed=42
    )

    node_sizes = [G.degree(node) * 100 for node in G.nodes()]
    edge_widths = [G[u][v]["weight"] * 0.05 for u, v in G.edges()]

    plt.figure(figsize=(15, 15))
    nx.draw(
        G,
        pos_spring,
        with_labels=True,
        node_size=node_sizes,
        width=edge_widths,
        font_family=plt.rcParams["font.family"],
        font_size=12,
        node_color="skyblue",
        edge_color="gray",
        alpha=0.8
    )
    plt.title("Kpop demon hunters keywords network", size=20)
    plt.axis("off")
    st.pyplot(plt.gcf())
    plt.close()

st.write(
    "시각화한 그래프를 보면 '2025', 'golden', 'ost' 등의 단어들이 이어져있는 것으로 보아 올해 음악적으로 큰 인기를 끌었음을 알 수 있다."
)


st.divider()

st.header("5️⃣plotly 그래프 시각화")

st.write(
    "키워드 등장 빈도 상위 항목을 Plotly 막대그래프로 시각화하였다. Plotly의 인터랙티브 기능을 활용하여 각 키워드의 빈도를 마우스로 확인할 수 있도록 시각화하였다. (plotly 바 차트 코드는 AI의 도움을 받아 작성하였습니다.)"
)

import pandas as pd
import plotly.express as px

kw_df_plotly = (
    pd.Series(text_kdh_clean2.split())
    .value_counts()
    .head(top_n)
    .reset_index()
)
kw_df_plotly.columns = ["keyword", "count"]

fig = px.bar(
    kw_df_plotly.sort_values("count", ascending=True),
    x="count",
    y="keyword",
    orientation="h",
    hover_data=["count"]
)

st.plotly_chart(fig, use_container_width=True)


st.write(
    "시각화 결과를 보면 미국과 한국과 같은 국가 키워드가 가장 높은 빈도를 보인다. 이는 국내와 해외에서 동시에 주목을 받고 있는 것을 보여준다. 또한 음악과 관련된 키워드들이 상위에 있는 것으로 보아 음악이 팬덤 형성에 기여하였음을 알 수 있다."
)

