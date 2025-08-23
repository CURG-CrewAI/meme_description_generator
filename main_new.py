# main.py
import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
from typing import Dict, Any
import json
from datetime import datetime

# 환경 변수 로드
load_dotenv()

def _get_api_key():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY in your .env")
    return api_key

def _check_serper_key():
    if not os.getenv("SERPER_API_KEY"):
        raise RuntimeError("Set SERPER_API_KEY in your .env (required by SerperDevTool)")

class MemeAgentCrew:
    def __init__(self):
        # 도구 설정
        _check_serper_key()
        self.search_tool = SerperDevTool()
        self.scrape_tool = ScrapeWebsiteTool()

        # CrewAI의 LLM으로 provider+model을 명시
        api_key = _get_api_key()
        self.gemini_pro = LLM(api_key=api_key, model="gemini/gemini-2.5-pro")
        self.gemini_flash = LLM(api_key=api_key, model="gemini/gemini-2.5-flash")
        self.gemini_flash_lite = LLM(api_key=api_key, model="gemini/gemini-2.5-flash-lite")
     
        # 키워드 검색 및 기사 URL 수집 에이전트
        self.search_agent = Agent(
            role='한국 뉴스 검색 전문가',
            goal='주어진 키워드의 최신 한국 뉴스 기사 URL과 제목을 수집',
            backstory=(
                """당신은 한국의 실시간 이슈와 뉴스에 대한 전문 지식을 가진 검색 전문가입니다.
                주어진 키워드로 가장 관련성 높고 신뢰할 수 있는 한국 뉴스 기사들을 찾아내는 것이 임무입니다.
                네이버 뉴스, 다음 뉴스, 연합뉴스, 조선일보, 중앙일보 등 주요 언론사의 기사를 우선적으로 검색합니다.
                검색할 때는 site:naver.com 또는 site:daum.net 같은 사이트 지정 검색을 활용합니다."""
            ),
            tools=[self.search_tool],
            verbose=True,
            llm=self.gemini_flash_lite
        )
       
        # 뉴스 기사에서 본문을 추출하는 에이전트
        self.extractor_agent = Agent(
            role='뉴스 본문 추출 전문가',
            goal='기사 URL에서 핵심 본문을 추출하고 요약',
            backstory=(
                """당신은 웹페이지에서 핵심 정보를 추출하는 전문가입니다.
                뉴스 기사의 제목, 본문, 핵심 내용을 정확하게 추출하여 
                다음 단계의 풍자글 작성에 필요한 정보를 제공합니다.
                광고나 불필요한 내용은 제외하고 순수 기사 내용만 추출합니다.
                특히 한국어 기사에서 중요한 사실과 배경을 파악하는 데 전문성을 가지고 있습니다."""
            ),
            tools=[self.scrape_tool],
            verbose=True,
            llm=self.gemini_flash
        )
    
        # 이슈에 대한 풍자글을 작성하는 에이전트
        self.writer_agent = Agent(
            role='밈코인 풍자글 작성 전문가',
            goal='수집된 뉴스 정보를 바탕으로 재미있고 창의적인 영어 밈코인 풍자글을 작성합니다.',
            backstory="""
            당신은 한국의 인터넷 문화와 밈에 정통한 창의적인 작가이면서, 
            글로벌 암호화폐 및 밈코인 트렌드에도 해박한 지식을 가지고 있습니다.
            한국 시사 이슈를 유머러스하게 풀어내며, 밈코인이라는 개념에 맞춰 
            재미있으면서도 적절한 선을 지키는 영어 풍자글을 작성하는 전문가입니다.
            
            작성 스타일:
            - 한국 이슈를 글로벌 관객이 이해할 수 있게 영어로 설명
            - 밈코인의 특성을 살린 창의적인 네이밍과 컨셉
            - 적절한 이모지와 해시태그 활용
            - 과도하지 않은 건전한 유머
            - 투자 주의사항을 반드시 포함
        
            ⚠️ 중요: 모든 결과물은 반드시 영어로 작성하세요.
            """,
            verbose=True,
            llm=self.gemini_pro
        )

        # 토크노믹스와 로드맵을 작성하는 에이전트
        self.tokenomics_agent = Agent(
            role='밈코인 토크노믹스 & 로드맵 설계자',
            goal='풍자글을 바탕으로 황당하고 재미있는 토크노믹스와 로드맵을 생성',
            backstory="""
            당신은 어이없어서 헛웃음이 나오는 밈코인 프로젝트를 잘 만드는 전문가입니다.
            실제 크립토 프로젝트처럼 그럴듯한 토크노믹스와 
            실현 불가능한 로드맵을 만들어 풍자와 유머를 극대화합니다.
        
            작성 스타일:
            - 실제 백서처럼 전문 용어 사용하되 말도 안 되는 내용
            - 파이 차트, 퍼센트 등 숫자를 활용한 가짜 데이터
            - Q1, Q2 같은 분기별 로드맵 (불가능한 목표들)
            - 진지한 톤으로 웃긴 내용 전달
        
            ⚠️ 중요: 모든 결과물은 반드시 영어로 작성하세요.
            """,
            verbose=True,
            llm=self.gemini_pro  
        )

        # 웹사이트 봇 전달용 JSON 변환 에이전트
        self.summary_agent = Agent(
            role='웹사이트 봇 연동용 JSON 변환 전문가',
            goal='풍자글과 토크노믹스를 웹사이트 봇이 사용할 수 있는 JSON 형식으로 변환',
            backstory="""
            당신은 다양한 데이터를 구조화된 JSON 형식으로 변환하는 전문가입니다.
            밈코인 프로젝트의 모든 정보를 웹사이트에서 사용하기 쉬운 형태로 정리하는 것이 임무입니다.
            각 섹션을 명확히 분리하고, 웹사이트에서 쉽게 파싱할 수 있도록 구조화합니다.
            """,
            verbose=True,
            llm=self.gemini_flash
        )

    def create_search_task(self, keyword: str, why_trending: str):
        context_term = (why_trending or "").split(".")[0]
        return Task(
            description=f"""
            주어진 키워드 '{keyword}'와 트렌딩 이유 '{why_trending}'를 바탕으로, 가장 정확하고 관련성 높은 최신 한국 뉴스 기사를 찾아야 합니다.

            단순히 키워드로만 검색하는 대신, 아래의 검색 전략을 사용하여 검색의 질을 높이세요.

            1.  **특정 뉴스 포털 타겟 검색:** `"{keyword}" site:news.naver.com` 또는 `"{keyword}" site:v.daum.net` 와 같이 신뢰도 높은 주요 포털을 직접 겨냥하여 검색합니다.
            2.  **컨텍스트 활용 검색:** 트렌딩 이유에서 핵심 단어를 뽑아 검색어에 조합합니다. 예: `"{keyword}" {context_term}`
            3.  **최신 키워드 검색:** `"{keyword}" 최신 뉴스` 또는 `"{keyword}" 공식입장` 과 같은 키워드를 조합하여 가장 최근 정보를 찾습니다.

            이 전략들을 종합적으로 활용하여, 신뢰할 수 있는 최신 기사 3~5개를 찾아 제목, URL, 간단한 요약을 수집해주세요.

            결과는 반드시 아래의 JSON 형식으로 정리해야 합니다:
            {{
            "keyword": "{keyword}",
            "articles": [
                {{
                    "title": "기사 제목",
                    "url": "기사 URL",
                    "snippet": "기사 요약",
                    "source": "언론사명"
                }}
            ]
        }}
        """,
        expected_output="신뢰도 높은 최신 기사 5~7개에 대한 정보가 담긴 JSON 형식의 문자열",
        agent=self.search_agent
    )
    
    def create_extraction_task(self):
        return Task(
            description="""
            이전 단계에서 수집된 기사 URL들에서 본문 내용을 추출하세요.
            
            추출 요구사항:
            1. 각 URL에 접속하여 기사 본문 추출
            2. 제목, 첫 2-3단락의 핵심 내용 중심으로 추출
            3. 광고, 관련기사, 댓글 등은 제외
            4. 핵심 사실과 맥락 정보만 정리
            5. 이슈의 배경과 현재 상황을 명확히 파악
            
            결과를 다음 형식으로 정리해주세요:
            {{
                "extracted_content": [
                    {{
                        "title": "기사 제목",
                        "content": "추출된 핵심 본문",
                        "key_points": ["핵심1", "핵심2", "핵심3"],
                        "background": "이슈 배경"
                    }}
                ]
            }}
            """,
            expected_output="JSON 형식의 추출된 기사 본문들",
            agent=self.extractor_agent
        )
    
    def create_satire_task(self, keyword: str):
        return Task(
            description=f"""
            키워드 '{keyword}'에 대해 수집된 정보를 바탕으로 영어 밈코인 풍자글을 작성하세요.
            
            풍자글 요구사항:
            1. 제목: 창의적인 밈코인 이름과 티커 ex)Sonnywood, $SONNY
            2. 본문: 이슈에 대한 재미있는 영어 설명 (100-250 단어)
            3. 특징: 해당 밈코인의 유머러스한 특징 1-4개 (영어)
            4. 주의사항: 풍자에 가까운 투자 주의사항 및 면책조항 (영어)
            5. 해시태그: 관련 영어 해시태그 5-7개
            
            톤앤매너:
            - 유머러스하지만 품격있게
            - 글로벌 관객이 이해할 수 있는 영어 표현
            - 이모지 적절히 사용
            - 아이러니와 풍자를 섞어 재구성하세요. 과장법, 비유, 말장난 등을 활용하세요.
            - 한국 문화적 맥락을 영어로 잘 설명
            - 생소한 한국의 이슈를 이해할 수 있도록 전달해야 함
            ⚠️ 중요: 모든 내용을 영어로 작성하세요.
            
            작성 형식:
            🚀 [COIN_NAME] Emergency Launch! 
            
            "[English Catchphrase]"
            
            [Korean issue explained in English with humor]
            
            ✨ Features:
            - Feature 1 in English
            - Feature 2 in English
            - Feature 3 in English
            
            ⚠️ Investment Warning: 
            [Disclaimer in English about this being entertainment only]
            
            #hashtag1 #hashtag2 #hashtag3 #hashtag4 #hashtag5


            작성 가이드(Few Shot):
            아래는 '배우의 사생활 확인 불가'라는 주제로 작성된 성공적인 풍자글의 작성 예시입니다. 
            이 예시의 창의적인 접근 방식을 학습하여 당신의 결과물에 적용하세요.

            주제:배우 정우성의 '사생활이라 확인 불가'라는 공식 입장
            훌륭한 Features 작성 예시:
            ✨ Features:
            - Non-Denial-Denial Protocol: Transactions are never officially confirmed, they just appear on the blockchain. We respect their privacy.
            - Plot Twist Burn Mechanism: A small percentage of the total supply is automatically burned every time a shocking new personal detail about a major public figure is "unable to be confirmed."
            - Stealth Wedding Rewards: Holders are airdropped bonus tokens when a long-term bachelor/bachelorette finally—and quietly—gets married without a press release.

            핵심 학습 포인트: 위 예시를 그대로 따라하지는 마십시오.
            대신, 풍자의 핵심 컨셉을 **'Protocol', 'Mechanism', 'Rewards', 'Tokenomics', 'Governance' 등 그럴듯한 블록체인 관련 기술 용어로 재치있게 포장**하여 유머를 극대화하는 것이 당신의 임무입니다. 주제에 가장 잘 맞는 용어를 창의적으로 선택하세요.
            예를 들어, 정치인을 풍자할 때는 Governance Model이나 Voting System이 더 어울릴 수 있고, 식품 관련 이슈에는 Supply Chain Tokenomics가 더욱 재미있을 수 있습니다.

            좋은 Investment Warning 작성 예시:
            ⚠️ A Note From Our Legal Team (Probably):
            Look, this is an emotional support token, not a financial instrument. Our entire whitepaper is a rumor someone started in a group chat. The liquidity pool is filled with pure, unadulterated speculation. Any resemblance to actual persons, living or dead, or actual events is purely coincidental and, frankly, a matter of their private life we cannot confirm. Buying this token will not make you rich, but it might make you laugh while you lose money on other, more serious-looking coins. This is not financial advice. This is performance art. DYOR (Do Your Own Research... on everything).

            핵심 학습 포인트: 위 예시를 그대로 따라하지는 마십시오.
            다만 위 예시처럼 경고문 전체를 마치 밈코인 개발자인 것처럼 작성하여, 경고하는 척하면서 재미있는 풍자를 이어나가는 전략은 권장합니다.

            """,
            expected_output="완성된 영어 밈코인 풍자글",
            agent=self.writer_agent
        )

    def create_tokenomics_task(self, keyword: str):
        return Task(
            description=f"""
            이전 단계(context)에서 생성된 '{keyword}' 밈코인 풍자글을 바탕으로 토크노믹스와 로드맵을 작성하세요.

            당신의 가장 중요한 임무는 **글의 일관성을 유지하는 것**입니다.
            풍자글 도입부에 제시된 `✨ Features` 항목들을, 당신이 작성할 `TOKENOMICS`의 `Special Features` 섹션에서 **반드시 가져와서 상세히 설명해야 합니다.** 절대 그 아이디어들을 무시하고 새로운 메커니즘을 만들지 마세요.

            💡 **워크플로우 가이드 (Your Workflow):**
            1.  컨텍스트로 받은 풍자글에서 `✨ Features` 섹션을 주의 깊게 읽으세요.
            2.  당신이 작성할 `TOKENOMICS`의 `Special Features` 섹션 제목으로, 해당 기능들의 이름을 그대로 사용하세요.
            3.  각 기능이 토크노믹스 관점에서 구체적으로 어떻게 작동하는지(예: 스테이킹 방식, 소각 방식, 보상 방식 등)를 창의적이고 재미있게 설명하세요.
            4.  만약 풍자글에 없던 새로운 메커니즘(예: 일반 소각)을 추가하고 싶다면, 기존 기능들을 모두 설명한 **후에** 덧붙이는 것은 괜찮습니다.
        
            📊 TOKENOMICS 요구사항:
            1. Total Supply: 황당한 숫자 (예: 420.69 trillion)
            2. Distribution:
            - Team: X% (웃긴 이유와 함께)
            - Marketing: X% (황당한 마케팅 계획)
            - Liquidity: X% 
            - Community: X% (말도 안 되는 커뮤니티 혜택)
            3. Special Features:
            **(필수) 풍자글의 `✨ Features`에서 가져온 아이디어들을 구체적으로 설명**
        
            🗺️ ROADMAP 요구사항:
            Q1 2025: [거짓말 같지만 약간 현실성 있는 목표]
            Q2 2025: [거의 불가능해보이는 황당한 목표]
            Q3 2025: [물리법칙 위반 수준의 완전히 불가능한 목표]
            Q4 2025: [우주 정복·시간여행 수준의 목표]
        
            형식:
            📊 TOKENOMICS
            ================
            Total Supply: [숫자]
        
            Distribution:
            • Team: X% - [웃긴 설명]
            • Marketing: X% - [황당한 계획]
            • Liquidity: X% - [과장된 설명]
            • Community: X% - [말도 안 되는 혜택]
        
            Special Features:
            • [풍자글의 `✨ Features`에서 가져온 아이디어들을 구체적으로 설명]
            • [풍자글의 `✨ Features`에서 가져온 아이디어들을 구체적으로 설명]
            • [풍자글의 `✨ Features`에서 가져온 아이디어들을 구체적으로 설명]
            
            🗺️ ROADMAP TO THE MOON (AND BEYOND)
            =====================================
            Q1 2025: [목표들]
            Q2 2025: [목표들]
            Q3 2025: [목표들]
            Q4 2025: [목표들]

            ⚠️ 모든 내용을 영어로 작성하고, 진지한 톤으로 웃긴 내용을 전달하세요. 이전 단계에서 생성된 풍자글과 비슷한 문체로 작성하세요.
            Pretend this is an official whitepaper presented to serious investors, but the content is absurd. Never break character.
            """,
            expected_output="토크노믹스와 로드맵이 포함된 완성된 보고서",
            agent=self.tokenomics_agent
        )

    def create_summary_task(self):
        return Task(
            description="""
            이전 단계에서 생성된 모든 정보(풍자글 + 토크노믹스 + 로드맵)를 웹사이트 봇이 사용할 수 있는 JSON 형식으로 변환하세요.

            JSON 구조 요구사항:
            {
                "name": "밈코인 전체 이름",
                "symbol": "밈코인 티커",
                "description": "본문 설명",
                "features": ["특징1", "특징2", "특징3"],
                "warning": "투자 경고문",
                "hashtags": ["해시태그1", "해시태그2", "해시태그3"],
                "tokenomics": {
                    "total_supply": "총 공급량",
                    "distribution": {
                        "team": "팀 할당 및 설명",
                        "marketing": "마케팅 할당 및 설명",
                        "liquidity": "유동성 할당 및 설명",
                        "community": "커뮤니티 할당 및 설명"
                    },
                    "special_features": ["특별 기능1", "특별 기능2"]
                },
                "roadmap": {
                    "Q1_2025": "1분기 목표",
                    "Q2_2025": "2분기 목표",
                    "Q3_2025": "3분기 목표",
                    "Q4_2025": "4분기 목표"
                }
            }

            JSON 구조 예시:

            주의사항:
            1. 모든 텍스트는 영어로 유지
            2. JSON 형식을 정확히 맞춰주세요
            3. 특수 문자나 이모지는 유지하되 JSON에 문제가 없도록 처리
            4. 각 섹션의 내용을 정확히 분류하여 배치
            """,
            expected_output="웹사이트 봇 전달용 JSON 데이터",
            agent=self.summary_agent
        )

    def run_satire_generation(self, keyword: str, why_trending: str):
        print(f"🤖 '{keyword}' 키워드로 전체 자동화 프로세스를 시작합니다...")
        print(f"📝 트렌딩 이유: {why_trending}\n")
        
        # 1. 모든 Task를 미리 정의합니다. 각 Task는 이전 Task의 결과를 context로 이어받습니다.
        search_task = self.create_search_task(keyword, why_trending)
        
        # context를 명시적으로 전달하도록 수정합니다.
        extraction_task = self.create_extraction_task()
        extraction_task.context = [search_task]
        
        satire_task = self.create_satire_task(keyword)
        satire_task.context = [extraction_task]
        
        tokenomics_task = self.create_tokenomics_task(keyword)
        tokenomics_task.context = [satire_task]
        
        summary_task = self.create_summary_task()
        summary_task.context = [tokenomics_task]

        # 2. Crew를 생성하고 모든 Task를 순서대로 실행합니다.
        crew = Crew(
            agents=[self.search_agent, self.extractor_agent, self.writer_agent, self.tokenomics_agent, self.summary_agent],
            tasks=[search_task, extraction_task, satire_task, tokenomics_task, summary_task],
            process=Process.sequential,
            verbose=True
        )
        
        try:
            # kickoff()는 모든 작업을 실행시키는 역할만 합니다. 최종 결과물은 아래에서 직접 추출합니다.
            crew.kickoff(inputs={"keyword": keyword, "why_trending": why_trending})
            
            # 3. 실행 완료 후, 필요한 결과물을 각 Task 객체에서 직접 추출합니다.
            # .output.raw를 사용하면 순수한 텍스트 결과물을 얻을 수 있습니다.
            satire_result = satire_task.output.raw
            tokenomics_result = tokenomics_task.output.raw
            summary_json = summary_task.output.raw
            
            # 4. 요구사항에 맞게 파일을 저장하고 데이터를 전달합니다.
            # 요구사항: 3번 에이전트의 결과물(풍자글)만 별도로 저장합니다.
            # 이것이 실제 런치패드에 올릴 '밈코인 설명'이 됩니다.
            
            # 출력 디렉토리 생성
            os.makedirs("./outputs", exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            satire_filename = f"./outputs/launchpad_description_{keyword}_{timestamp}.txt"
            
            with open(satire_filename, "w", encoding="utf-8") as f:
                f.write(satire_result)
            print(f"\n✅ 런치패드용 설명 파일 저장 완료: '{satire_filename}'")
            
            # (선택) 만약 전체 리포트(풍자글 + 토크노믹스)도 저장하고 싶다면 아래 주석을 해제하세요.
            # full_report_filename = f"./outputs/full_report_{keyword}_{timestamp}.txt"
            # with open(full_report_filename, "w", encoding="utf-8") as f:
            #     f.write(f"{satire_result}\n\n{tokenomics_result}")
            # print(f"✅ 전체 리포트 파일 저장 완료: '{full_report_filename}'")
            
            # 5번 에이전트의 결과물(JSON)은 파일로 저장하지 않고,
            # 웹사이트 봇에게 전달하기 위해 함수의 최종 결과로 반환합니다.
            print(f"✅ 웹사이트 봇에게 전달할 JSON 데이터 생성 완료.")
            
            return {
                'json_data': summary_json,
                'satire_filename': satire_filename,
                'satire_content': satire_result
            }
            
        except Exception as e:
            print(f"❌ 에러 발생: {e}")
            # 부분적으로라도 완성된 결과가 있다면 반환
            try:
                partial_results = self._get_partial_results([search_task, extraction_task, satire_task, tokenomics_task, summary_task])
                return partial_results if partial_results else None
            except:
                return None

    def _get_partial_results(self, tasks):
        """에러 발생 시 부분적으로라도 완성된 결과를 반환"""
        results = {}
        
        # 각 task의 output이 있는지 확인
        task_names = ['search', 'extraction', 'satire', 'tokenomics', 'summary']
        for i, task in enumerate(tasks):
            try:
                if hasattr(task, 'output') and task.output:
                    results[task_names[i]] = task.output.raw
            except:
                continue
        
        # 최소한 풍자글이라도 완성되었다면 부분 결과 반환
        if 'satire' in results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            try:
                os.makedirs("./outputs", exist_ok=True)
                satire_filename = f"./outputs/partial_launchpad_description_{timestamp}.txt"
                with open(satire_filename, "w", encoding="utf-8") as f:
                    f.write(results['satire'])
                print(f"⚠️ 부분 완성: 풍자글만 저장됨 '{satire_filename}'")
                
                return {
                    'json_data': results.get('summary', None),
                    'satire_filename': satire_filename,
                    'satire_content': results['satire'],
                    'partial': True
                }
            except:
                pass
        
        return None

# 팀 연동을 위한 함수
def generate_satire_for_team(input_data: dict) -> dict:
    """
    팀원 에이전트와 연동하기 위한 함수
    
    Args:
        input_data (dict): {
            "keyword": str, 
            "why_trending": str
        }
    
    Returns:
        dict: {
            'json_data': str,           # 웹사이트 봇용 JSON
            'satire_filename': str,     # 저장된 풍자글 파일 경로
            'satire_content': str,      # 풍자글 내용
            'partial': bool             # 부분 완성 여부 (optional)
        }
    """
    meme_crew = MemeAgentCrew()
    
    keyword = input_data.get("keyword", "")
    why_trending = input_data.get("why_trending", "")
    
    if not keyword:
        return {"error": "키워드가 제공되지 않았습니다."}
    
    try:
        result = meme_crew.run_satire_generation(keyword, why_trending)
        return result if result else {"error": "풍자글 생성에 실패했습니다."}
    except Exception as e:
        return {"error": f"풍자글 생성 중 오류 발생: {str(e)}"}

# 메인 실행 함수
def main():
    """테스트용 메인 함수"""
    meme_crew = MemeAgentCrew()
    
    # 팀원 에이전트로부터 받은 입력 예시
    keyword = "손흥민"
    why_trending = "손흥민 is a renowned South Korean football player. Recent reports indicate he may be transferring from Tottenham to LA FC."
    
    result = meme_crew.run_satire_generation(keyword, why_trending)
    
    if result and 'error' not in result:
        print("\n" + "="*50)
        print("🎉 생성 완료!")
        print("="*50)
        
        print("📄 저장된 파일:", result.get('satire_filename'))
        print("🌐 웹사이트 봇 전달용 JSON:")
        print(result.get('json_data'))
        
        if result.get('partial'):
            print("⚠️ 주의: 부분적으로만 완성되었습니다.")
    else:
        print("❌ 생성에 실패했습니다.")
        if result and 'error' in result:
            print(f"에러: {result['error']}")

# 간단한 테스트 함수
def quick_test():
    """빠른 테스트용 함수"""
    print("🧪 빠른 테스트 시작...")
    
    # 간단한 입력으로 테스트
    test_data = {
        "keyword": "정우성",
        "why_trending": "Famous Korean actor trending due to rumors"
    }
    
    result = generate_satire_for_team(test_data)
    
    print("\n📝 테스트 결과:")
    if 'error' in result:
        print(f"❌ {result['error']}")
    else:
        print(f"✅ 성공!")
        print(f"📄 파일: {result.get('satire_filename')}")
        print(f"🌐 JSON 데이터: {result.get('json_data')[:200]}..." if result.get('json_data') else "JSON 없음")

if __name__ == "__main__":
    # 실행 방법 선택
    print("실행 모드를 선택하세요:")
    print("1. 전체 테스트 (main)")
    print("2. 빠른 테스트 (quick_test)")
    
    choice = input("선택 (1 또는 2): ").strip()
    
    if choice == "1":
        main()
    elif choice == "2":
        quick_test()
    else:
        print("잘못된 선택입니다. 전체 테스트를 실행합니다.")
        main()