'## 🆎 : 일반 텍스트'

import streamlit as st

st.title('제목 : st.title()')
st.header('헤더 : st.header()')
st.subheader('서브헤더 : st.subheader()')
st.text('본문 텍스트 : st.text()')
st.markdown('## 마크다운 : st.markdown()')
st.caption('캡션(작고 흐린 글씨로 표현됨) : st.caption()')

'# 🆎 : st.write()'
st.write('# 마크다운 H1 : st.write()')
st.write('### 마크다운 H3 : st.write()')
st.write('')  # 빈 줄 추가


'# 🆎 : 색상이 있는 텍스트'
st.write(':red[빨간색 텍스트]')
st.write(':blue[파란색 텍스트]')


'### 코드 블록: st.code()'
st.code('print("Hello, World!")', language='python', line_numbers=True)

'### 코드+결과: st.echo()'
with st.echo():
    # 이 블록의 코드와 결과를 출력
    name = 'Chunghun Ha'
    st.write("Hello, Streamlit!", name)

'### Latex 수식 작성: st.latex()'
st.latex(r'\int_a^b f(x)dx')

st.divider()  # 👉 구분선



## 📌 텍스트 · 미디어 (`텍스트_미디어.py`)


'# 🎥 : 미디어 삽입'

'### :orange[이미지: st.image()]'
st.image("F:/Users/김 연호/Desktop/학교/2025-2/데이터시각화/streamlit/data/python.png", caption="파이썬 로고", use_container_width=True)

'### :orange[오디오: st.audio()]'
#st.audio("./data/After_You.mp3", format="audio/mpeg", loop=True)

'### :orange[동영상: st.video()]'
#st.video("./data/stars.mp4", format="video/mp4", loop=True)
st.video("https://www.youtube.com/watch?v=dQw4w9WgXcQ")  # YouTube 링크


'# 📊 : 콜아웃'

'### :orange[정보: st.info()]'
st.info('This is a purely informational message', icon="ℹ️")

'### :orange[경고: st.warning()]'
st.warning('This is a warning message', icon="⚠️")

'### :orange[에러: st.error()]'
st.error('This is an error message', icon="⛔")

'### :orange[성공: st.success()]'
st.success('This is a success message', icon="✅")
