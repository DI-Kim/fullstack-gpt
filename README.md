- AI 웹 서비스 (6종) : DocumentGPT, PrivateGPT, QuizGPT, SiteGPT, MeetingGPT, InvestorGPT
- ChatGPT 플러그인 (1종) : ChefGPT
- 활용하는 패키지 : Langchain, GPT-4, Whisper, FastAPI, Streamlit, Pinecone, Hugging Face… and more!

### DocumentGPT

법률. 의학 등 어려운 용어로 가득한 각종 문서. AI로 빠르게 파악하고 싶다면?

AI로 신속하고 정확하게 문서 내용을 파악하고 정리한 뒤, 필요한 부분만 쏙쏙 골라내어 사용하세요. DocumentGPT 챗봇을 사용하면, AI가 문서(.txt, .pdf, .docx 등)를 꼼꼼하게 읽고, 해당 문서에 관한 질문에 척척 답변해 줍니다.

### PrivateGPT

회사 기밀이 유출될까 걱정된다면? 이제 나만이 볼 수 있는 비공개 GPT를 만들어 활용하세요!

DocumentGPT와 비슷하지만 로컬 언어 모델을 사용해 비공개 데이터를 다루기에 적합한 챗봇입니다. 데이터는 컴퓨터에 보관되므로 오프라인에서도 사용할 수 있습니다. 유출 걱정 없이 필요한 데이터를 PrivateGPT에 맡기고 업무 생산성을 높일 수 있어요.

### QuizGPT

암기해야 할 내용을 효율적으로 학습하고 싶다면?

문서나 위키피디아 등 학습이 필요한 컨텐츠를 AI에게 학습시키면, 이를 기반으로 퀴즈를 생성해 주는 앱입니다. 번거로운 과정을 최소화하고 학습 효율을 극대화할 수 있어, 특히 시험이나 단기간 고효율 학습이 필요할 때 매우 유용하게 사용할 수 있어요.

### SiteGPT

자주 묻는 질문 때문에 CS 직원을 채용...? SiteGPT로 비용을 2배 절감해 봅시다.

웹사이트를 스크랩하여 콘텐츠를 수집하고, 해당 출처를 인용하여 관련 질문에 답변하는 챗봇입니다. 고객 응대의 대부분을 차지하는 단순 정보 안내에 들이는 시간을 획기적으로 줄일 수 있고, 고객 또한 CS직원의 근무 시간에 구애받지 않고 정확한 정보를 빠르게 전달받을 수 있습니다.

### MeetingGPT

이제 회의록 정리는 MeetingGPT에게 맡기세요!

회의 영상 내용을 토대로 오디오 추출, 콘텐츠를 수집하여 회의록을 요약 및 작성해 주는 앱입니다. 회의 내용을 기록하느라 회의에 제대로 참석하지 못하는 일을 방지할 수 있고, 관련 질의응답도 가능해 단순한 기록보다 훨씬 더 효율적으로 회의록을 관리하고 활용할 수 있습니다.

### InvestorGPT

AI가 자료 조사도 알아서 척척 해 줍니다.

인터넷을 검색하고 타사 API를 사용할 수 있는 자율 에이전트입니다. 회사, 주가 및 재무제표를 조사하여 재무에 대한 인사이트를 제공할 수 있습니다. 또한 알아서 데이터베이스를 수집하기 때문에 직접 SQL 쿼리를 작성할 필요가 없고, 해당 내용에 대한 질의응답도 얼마든지 가능합니다.

### ChefGPT

요즘 핫한 ChatGPT 플러그인? 직접 구현해 봐요!

유저가 ChatGPT 플러그인 스토어에서 설치할 수 있는 ChatGPT 플러그인입니다. 이 플러그인을 통해 유저는 ChatGPT 인터페이스에서 바로 레시피를 검색하고 조리법을 얻을 수 있습니다. 또한 ChatGPT 플러그인에서 OAuth 인증을 구현하는 방법에 대해서도 배웁니다.

---

### Configure | virtual environment

1. `python3 -m venv ./env`
2. `source <venv>/bin/activate` | \<venv> is env folder you made
3. make requirements.txt for matching same version of libraries
4. `pip install -r requirements.txt`

#### exit virtual environment `deactivate`

---

### Streamlit

run streamlit

1. `source env/bin/activate`
2. `streamlit run <Filename>` | ex) `streamlit run Home.py`
