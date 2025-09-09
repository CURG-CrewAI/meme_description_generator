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
            role='Korean News Search Expert',
            goal='Collect URLs and titles of the latest Korean news articles for given keywords',
            backstory=(
                """You are a search expert with specialized knowledge of real-time Korean issues and news.
                Your mission is to find the most relevant and reliable Korean news articles for given keywords.
                You prioritize searching articles from major news outlets such as Naver News, Daum News, 
                Yonhap News, Chosun Ilbo, and JoongAng Ilbo.
                When searching, you utilize site-specific searches like site:naver.com or site:daum.net."""
            ),
            tools=[self.search_tool],
            verbose=True,
            llm=self.gemini_flash_lite
        )
       
        # 뉴스 기사에서 본문을 추출하는 에이전트
        self.extractor_agent = Agent(
            role='News Content Extraction Expert',
            goal='Extract and summarize core content from article URLs',
            backstory=(
                """You are an expert at extracting key information from web pages.
                You accurately extract article titles, body content, and key information to provide 
                necessary data for subsequent AI Agents' article summarization, translation, and satirical writing tasks.
                You exclude advertisements and unnecessary content, focusing only on pure article content.
                You specialize in identifying important facts and background information from Korean-language articles."""
            ),
            tools=[self.scrape_tool],
            verbose=True,
            llm=self.gemini_flash_lite
        )
    
        # 요약한 기사 본문을 취합하여 요약하고 영어로 번역하는 에이전트
        self.translator_agent = Agent(
            role='Article Summarization and English Translation Expert',
            goal='Summarize Korean articles to appropriate length and translate them into English',
            backstory=(
                """You are an expert at compiling, summarizing, and translating extracted Korean news 
                in a way that global readers can easily understand.
                You translate complex Korean issues clearly with context and background 
                so that foreigners can easily comprehend them."""
            ),
            verbose=True,                             
            llm=self.gemini_flash
        )

        # 웹사이트용 설명 생성 에이전트
        self.description_agent = Agent(
            role='Website Description Generator',
            goal='Create concise 3-4 sentence descriptions for website tone setting',
            backstory=("""
            You are an expert at creating brief, impactful descriptions that set the overall tone for websites.
            You specialize in distilling complex Korean issues into clear, neutral explanations 
            that foreign readers can quickly understand and that determine the website's atmosphere.
            """),
            verbose=True,
            llm=self.gemini_flash
        )

        # 이슈에 대한 풍자글을 작성하는 에이전트
        self.writer_agent = Agent(
            role='Memecoin Satirical Content Writing Expert',
            goal='Create funny and creative English memecoin satirical content based on collected news information',
            backstory="""
            You are a creative writer well-versed in Korean internet culture and memes, 
            with extensive knowledge of global cryptocurrency and memecoin trends.
            You specialize in writing English satirical content that humorously presents Korean current affairs 
            while maintaining appropriate boundaries and fitting the memecoin concept.

            Writing Style:
            - Explain Korean issues in English that global audiences can understand
            - Creative naming and concepts that leverage memecoin characteristics
            - Appropriate use of emojis and hashtags
            - Healthy humor that is not excessive

            ⚠️ Important: All content must be written in English.
            """,
            verbose=True,
            llm=self.gemini_pro
        )

        # 토크노믹스와 로드맵을 작성하는 에이전트
        self.tokenomics_agent = Agent(
            role='Memecoin Tokenomics & Roadmap Designer',
            goal='Generate absurd and funny tokenomics and roadmaps based on satirical content',
            backstory="""
            You are an expert at creating memecoin projects that are so absurd they make people laugh out loud.
            You create plausible-sounding tokenomics and impossible roadmaps that maximize satire and humor,
            just like actual crypto projects.

            Writing Style:
            - Use professional terminology like actual whitepapers but with nonsensical content
            - Utilize fake data with pie charts, percentages, and numbers
            - Quarterly roadmaps like Q1, Q2 with impossible goals
            - Deliver funny content with a serious tone

            ⚠️ Important: All content must be written in English.
            """,
            verbose=True,
            llm=self.gemini_pro  
        )

        # 웹사이트 봇 전달용 JSON 변환 에이전트
        self.summary_agent = Agent(
            role='Website Bot Integration JSON Conversion Expert',
            goal='Convert satirical content and tokenomics into JSON format usable by website bots',
            backstory="""
            You are an expert at converting various data into structured JSON formats.
            Your mission is to organize all memecoin project information into forms that are easy to use on websites.
            You clearly separate each section and structure them so they can be easily parsed by websites.
            """,
            verbose=True,
            llm=self.gemini_flash
        )

    def create_search_task(self, keyword: str, why_trending: str):
        context_term = (why_trending or "").split(".")[0]
        return Task(
            description=f"""
            Based on the given keyword '{keyword}' and trending reason '{why_trending}', you need to find the most accurate and relevant latest Korean news articles.

            Instead of simply searching by keywords alone, use the following search strategies to improve search quality:

            1. **Targeted News Portal Search:** Directly target high-credibility major portals like `"{keyword}" site:news.naver.com` or `"{keyword}" site:v.daum.net`.
            2. **Context-Driven Search:** Extract key terms from the trending reason and combine them with the search term. Example: `"{keyword}" {context_term}`
            3. **Latest News Search:** Combine keywords like `"{keyword}" 최신 뉴스` or `"{keyword}" 공식입장` to find the most recent information.

            Using these strategies comprehensively, collect 3-5 reliable latest articles with titles, URLs, and brief summaries.

            Results must be organized in the following JSON format:
            {{
            "keyword": "{keyword}",
            "articles": [
                {{
                    "title": "Article Title",
                    "url": "Article URL", 
                    "snippet": "Article Summary",
                    "source": "Media Outlet Name"
                }}
            ]
        }}
        """,
        expected_output="JSON format string containing information about 5-7 reliable latest articles",
        agent=self.search_agent
    )
    
    def create_extraction_task(self):
        return Task(
            description="""
            Extract body content from the article URLs collected in the previous step.
            
            Extraction Requirements:
            1. Access each URL and extract article body content
            2. Focus on extracting titles and key content from the first 2-3 paragraphs
            3. Exclude advertisements, related articles, comments, etc.
            4. Organize only key facts and contextual information
            5. Clearly identify issue background and current situation
            6. Skip inaccessible or failed extraction URLs and continue processing
            7. Write results using only successfully extracted articles
            8. Consider task complete if at least 1 article is successfully extracted
            9. If text contains repeated sentences more than 5 times, skip that article
            10. If Unicode decoding errors occur, skip that specific article and continue

            Error Handling:
            - Exclude only the specific article for URL access failures, scraping blocks, etc.
            - Log failed articles but do not stop the entire operation
            - Compose extracted_content array only with extractable articles
            
            Organize results in the following format:
            {{
                "extracted_content": [
                    {{
                        "title": "Article Title",
                        "url": "Article URL",
                        "content": "Extracted Key Body Content",
                        "key_points": ["Key Point 1", "Key Point 2", "Key Point 3"],
                        "background": "Issue Background",
                        "source": "Media Outlet Name"
                    }}
                ]
            }}
            """,
            expected_output="Extracted article body content in JSON format",
            agent=self.extractor_agent
        )
    
    def create_translation_task(self):
        return Task(
            description="""
            Compile, summarize, and translate the extracted JSON format article content from the previous step into English.

            Processing Steps:
            1. Comprehensively analyze all article content in the extracted_content array
            2. Identify common themes and key content from multiple articles
            3. Extract 2-3 key keywords suitable for image generation (key person names, places, core concepts, key objects, etc.)
            - Examples: "Son Heung-min, LAFC Stadium, Soccer Ball" or "Lee Jae-myung, 250,000 won"
            - Select keywords focusing on specific, searchable nouns
            - Prioritize visually representable elements over abstract concepts
            4. Write content within 200-300 words total
            5. Include actual URLs in the Sources section
            6. Include Korean cultural context and background explanations for foreign understanding
            7. Translate everything into English considering foreign readers

            Output Format:
            # Korean Issue Summary: [Issue Title]

            **Key Visual Keywords:** ["Keyword1","Keyword2","Keyword3"]

            **Background:** 
            [Korean cultural/social context explanation]

            **What Happened:** 
            [Current situation and major events]

            **Key Points:**
            - [Key Point 1]
            - [Key Point 2] 
            - [Key Point 3]

            **Why It Matters:**
            [Why this issue is important and noteworthy]

            This summary serves as material to help foreigners interested in memecoins understand the background of Korean issues.

            **Sources:**
            [Media Outlet] - [Article Title] - [URL]

            Write media outlet names, article titles, and source URLs in English.
            Must include all sources used for compilation.
            Provide 2-3 or more different media outlet sources when possible.
            List as many as the actual number of referenced articles.

            Reference the following examples:
            - Yonhap News Agency - "Lawmaker Lee Under Investigation for Stock Trading" - https://...
            - JTBC News - "45 People Questioned in Nominee Trading Case" - https://...
            - KBS News - "Police Analyze Financial Records in Political Scandal" - https://...

            Note: Use actual URLs, not [URL Placeholder].
            """,
            expected_output="Korean issue comprehensive summary translated into English",
            agent=self.translator_agent
        )

    def create_description_task(self):
        return Task(
            description="""
            Based on the Korean issue summary from the previous step, create a concise 3-4 sentence description that sets the overall tone for the website.

            Requirements:
            - Length: 3-4 sentences, 80-120 words
            - Purpose: Core summary for determining website atmosphere
            - Include: Who, what, why it's problematic (briefly)
            - Exclude: Detailed background explanations, Key Points listing
            - Tone: Information-focused neutral tone
            - Target: Foreign readers unfamiliar with Korean issues

            Writing Guide:
            1. First sentence: Simply explain who did what
            2. Second sentence: Core issues or problems
            3. Final sentence: Why this issue is important

            Example: "South Korean lawmaker Lee Chun-seok is under investigation for alleged stock trading using nominee accounts worth 1 billion won. The amount significantly exceeds his declared assets, raising questions about the source of funds. This case highlights ongoing concerns about financial transparency in Korean politics."

            Reference the example but do not copy it directly.
            """,
            expected_output="3-4 sentence concise description for website tone setting",
            agent=self.description_agent
    )

    def create_satire_task(self, keyword: str):
        return Task(
            description=f"""
            Write English memecoin satirical content based on the collected information about keyword '{keyword}'.
            
            Satirical Content Requirements:
            1. Title: Creative memecoin name and ticker ex) Sonnywood, $SONNY
            2. Body: Funny English explanation of the issue (100-250 words)
            3. Features: 1-4 humorous features of the memecoin (in English)
            4. Hashtags: 5-7 related English hashtags

            Tone and Style:
            - Humorous but dignified
            - English expressions that global audiences can understand
            - Use emojis appropriately
            - Mix irony and satire through exaggeration, metaphors, wordplay, etc.
            - Explain Korean cultural context well in English
            - Make unfamiliar Korean issues understandable
            ⚠️ Important: Write all content in English.
            
            Writing Format:
            🚀 [COIN_NAME] 
            
            "[English Catchphrase]"
            
            [Korean issue explained in English with humor]
            
            ✨ Features:
            - Feature 1 in English
            - Feature 2 in English
            - Feature 3 in English
            
            #hashtag1 #hashtag2 #hashtag3 #hashtag4 #hashtag5

            Writing Guide (Few Shot):
            Below is an example of successful satirical writing on the theme of 'actor's private life cannot be confirmed.' 
            Learn from this example's creative approach and apply it to your results.

            Topic: Actor Jung Woo-sung's official statement that his 'private life cannot be confirmed'
            Excellent Features writing example:
            ✨ Features:
            - Non-Denial-Denial Protocol: Transactions are never officially confirmed, they just appear on the blockchain. We respect their privacy.
            - Plot Twist Burn Mechanism: A small percentage of the total supply is automatically burned every time a shocking new personal detail about a major public figure is "unable to be confirmed."
            - Stealth Wedding Rewards: Holders are airdropped bonus tokens when a long-term bachelor/bachelorette finally—and quietly—gets married without a press release.

            Key Learning Points: Do not copy the above example directly.
            Instead, your mission is to maximize humor by cleverly packaging the core satirical concept with **plausible blockchain-related technical terms like 'Protocol', 'Mechanism', 'Rewards', 'Tokenomics', 'Governance'**. Creatively choose terms that best fit your topic.
            For example, when satirizing politicians, Governance Model or Voting System might be more suitable, while food-related issues could be funnier with Supply Chain Tokenomics.
            """,
            expected_output="Complete English memecoin satirical content",
            agent=self.writer_agent
        )

    def create_tokenomics_task(self, keyword: str):
        return Task(
            description=f"""
            Write tokenomics and roadmap based on the '{keyword}' memecoin satirical content generated in the previous step (context).

            Your most important mission is **maintaining content consistency**.
            You must **bring the `✨ Features` items presented in the satirical content introduction to your `TOKENOMICS` `Special Features` section and explain them in detail.** Never ignore these ideas and create new mechanisms.

            💡 **Workflow Guide:**
            1. Carefully read the `✨ Features` section from the satirical content received as context.
            2. Use the names of those features exactly as section titles for your `TOKENOMICS` `Special Features` section.
            3. Creatively and entertainingly explain how each feature works specifically from a tokenomics perspective (e.g., staking methods, burning methods, reward methods, etc.).
            4. If you want to add new mechanisms not in the satirical content (e.g., general burning), it's acceptable to add them **after** explaining all existing features.

            📊 TOKENOMICS Requirements:
            1. Distribution:
            - Team: 0-1% (with funny reasons, team allocation should preferably be minimal)
            - Marketing: X% (absurd marketing plans)
            - Liquidity: X% 
            - Community: X% (ridiculous community benefits)
            2. Special Features:
            **(Required) Detailed explanation of ideas brought from the satirical content's `✨ Features`**

            🗺️ ROADMAP Requirements:
            Q1 2025: [Unbelievable but somewhat realistic goals]
            Q2 2025: [Nearly impossible absurd goals]
            Q3 2025: [Completely impossible goals that violate physical laws]
            Q4 2025: [Universe conquest/time travel level goals]

            Format:
            📊 TOKENOMICS
            ================
            Distribution:
            - Team: 0-1% - [Funny explanation]
            - Marketing: X% - [Absurd plans]
            - Liquidity: X% - [Exaggerated explanation]
            - Community: X% - [Ridiculous benefits]

            Special Features:
            - [Detailed explanation of ideas brought from satirical content's `✨ Features`]
            - [Detailed explanation of ideas brought from satirical content's `✨ Features`]
            - [Detailed explanation of ideas brought from satirical content's `✨ Features`]

            🗺️ ROADMAP TO THE MOON (AND BEYOND)
            =====================================
            Q1 2025: [Goals]
            Q2 2025: [Goals]
            Q3 2025: [Goals]
            Q4 2025: [Goals]

            ⚠️ Write all content in English and deliver funny content with a serious tone. Write in a similar style to the satirical content generated in the previous step.
            Pretend this is an official whitepaper presented to serious investors, but the content is absurd. Never break character.
            """,
            expected_output="Complete report including tokenomics and roadmap",
            agent=self.tokenomics_agent
        )

    def create_summary_task(self):
        return Task(
            description="""
            Convert all information generated in previous steps (article summaries, satirical content, etc.) into JSON format usable by website bots.

            JSON Structure Requirements:
            {
                "name": "Full memecoin name",
                "symbol": "Memecoin ticker",
                "description": "Use the result generated by description_task as is",
                "korean_issue_summary": {
                    "title": "Issue title",
                    "background": "Korean cultural/social context explanation",
                    "what_happened": "Current situation and major events",
                    "key_points": ["Key Point 1", "Key Point 2", "Key Point 3"],
                    "why_it_matters": "Why this issue is important and noteworthy"
                },
                "sources": [
                    {
                        "outlet": "Media outlet name",
                        "title": "Article title",
                        "url": "Article URL"
                    }
                ],
                "hashtags": ["Hashtag1", "Hashtag2", "Hashtag3"],
            }

            JSON Structure Example:

            Guidelines:
            1. Keep all text in English
            2. Ensure accurate JSON formatting
            3. Maintain special characters and emojis but ensure they don't break JSON structure
            4. Accurately classify and place content in each section
            5. For the "description" field, use the 3-4 sentence summary from the description_task result as is
            """,
            expected_output="JSON data for website bot delivery",
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
        
        translation_task = self.create_translation_task()
        translation_task.context = [extraction_task]

        description_task = self.create_description_task()
        description_task.context = [translation_task]

        satire_task = self.create_satire_task(keyword)
        satire_task.context = [extraction_task]
        
        tokenomics_task = self.create_tokenomics_task(keyword)
        tokenomics_task.context = [satire_task]
        
        summary_task = self.create_summary_task()
        summary_task.context = [translation_task, description_task, tokenomics_task]

        # 2. Crew를 생성하고 모든 Task를 순서대로 실행합니다.
        crew = Crew(
            agents=[self.search_agent, self.extractor_agent, self.translator_agent, self.description_agent, self.writer_agent, self.tokenomics_agent, self.summary_agent],
            tasks=[search_task, extraction_task, translation_task, description_task, satire_task, tokenomics_task, summary_task],
            process=Process.sequential,
            verbose=True
        )
        
        try:
            # kickoff()는 모든 작업을 실행시키는 역할만 합니다. 최종 결과물은 아래에서 직접 추출합니다.
            crew.kickoff(inputs={"keyword": keyword, "why_trending": why_trending})
            
            # 3. 실행 완료 후, 필요한 결과물을 각 Task 객체에서 직접 추출합니다.
            # .output.raw를 사용하면 순수한 텍스트 결과물을 얻을 수 있습니다.
            translation_result = translation_task.output.raw
            description_result = description_task.output.raw
            satire_result = satire_task.output.raw
            tokenomics_result = tokenomics_task.output.raw
            summary_json = summary_task.output.raw
            
            # 4. 요구사항에 맞게 파일을 저장하고 데이터를 전달합니다.
            # 요구사항: 3번 에이전트의 결과물(풍자글)만 별도로 저장합니다.
            # 이것이 실제 런치패드에 올릴 '밈코인 설명'이 됩니다.
            
            # 출력 디렉토리 생성
            os.makedirs("./outputs", exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            
            translation_filename = f"./outputs/korean_issue_summary_{keyword}_{timestamp}.txt"
            with open(translation_filename, "w", encoding="utf-8") as f:
                f.write(translation_result)
            print(f"✅ 한국 이슈 요약본 저장 완료: '{translation_filename}'")

            description_filename = f"./outputs/website_description_{keyword}_{timestamp}.txt"
            with open(description_filename, "w", encoding="utf-8") as f:
                f.write(description_result)
            print(f"✅ 웹사이트 설명 저장 완료: '{description_filename}'")

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
                partial_results = self._get_partial_results([search_task, extraction_task, translation_task, description_task, satire_task, tokenomics_task, summary_task])
                return partial_results if partial_results else None
            except:
                return None

    def _get_partial_results(self, tasks):
        """에러 발생 시 부분적으로라도 완성된 결과를 반환"""
        results = {}

        
        # 각 task의 output이 있는지 확인
        task_names = ['search', 'extraction', 'translation', 'description', 'satire', 'tokenomics', 'summary']
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
    keyword = "시드니 스위니"
    why_trending = "Sydney Sweeney is an American actress. She's trending due to controversy surrounding an advertisement for American Eagle jeans."
    
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