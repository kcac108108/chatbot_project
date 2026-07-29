# 교부기관 이력관리시스템 지능형 검색 기반 챗봇 구축 PoC
## 개발 요구사항 정의서 (Claude Code 개발 지시용)

작성기준일: 2026-07-24 / 작성: 통계서비스팀

---

## 1. 프로젝트 개요

- 이력관리시스템의 4개 메뉴 데이터를 RAG(검색 증강 생성) 방식으로 적재하고, 자연어 질문으로 관련 이력을 검색·요약 답변하는 챗봇을 구축함
- PoC 범위: 4개 메뉴의 DB 정형 데이터 (1단계). 첨부파일 내용 청킹은 2단계로 분리함
- 전체 예상 청크 수: 약 914건 이상 (메뉴1 약 836 + 메뉴2 8 + 메뉴3 13 + 메뉴4 57 + 기관 소개 청크)

### 1.1 아키텍처 (상용 API 방식)

```
[적재 배치] Oracle 조회 → 청크 조립 → OpenAI 임베딩 → 벡터DB 저장
[챗봇 API] 질문 → 질문 임베딩 → 벡터 검색(+메타 필터) → LLM 답변 생성 → 출처 포함 응답
[웹 화면]  질문 입력 / 답변 + 근거(출처) 표시
```

- 임베딩 모델: OpenAI `text-embedding-3-large` (문서 저장·질문 검색 동일 모델 사용 필수)
- LLM: OpenAI GPT 계열 (temperature 0.2 이하 고정, 근거 기반 답변 프롬프트)
- 벡터 DB: Qdrant (Docker), 단일 컬렉션 + `데이터유형` 메타데이터로 구분
- API 서버: FastAPI (기존 steel-classify와 동일 구조)

### 1.2 개발/운영 환경 제약 (중요)

- 운영 서버: 폐쇄망 Rocky Linux, Docker 구동. 외부 API(OpenAI) 통신 경로만 예외 개방됨
- 개발: 외부 PC에서 Claude Code로 개발 → Docker 이미지 반입 → 접속정보(환경변수)만 교체하여 구동
- **실데이터는 개발 환경에서 절대 사용 금지. 전 과정 더미 데이터로 개발·테스트함**
- 모든 pip 패키지·리소스는 빌드 시점에 이미지에 내장 (런타임 외부 다운로드 금지)
- DB 접속·API 키·경로는 전부 환경변수로 분리 (개발/운영 전환 대비)
- 운영 DB는 DB링크(@STATWEB) 경유 조회임. 개발 로컬 DB에는 링크 없이 동일 테이블명으로 구성

### 1.3 개발 환경 구축 요구

1. 로컬 Oracle XE (Docker 컨테이너)
2. 대상 테이블 DDL 실행 (운영과 동일 구조, @STATWEB 링크 제거판)
3. 더미 데이터 생성기 작성:
   - 조인 정합성 유지 (기관→이력→첨부의 키 연결이 실제처럼 맞물릴 것)
   - 4가지 케이스 모두 생성: ①메인만 ②메인+메인첨부 ③메인+이력+이력첨부 ④전부
   - 첨부 0~7개 분포, NULL 컬럼 분포, 서술형 컬럼은 한글 100~500자 유사 텍스트
   - 특수문자·줄바꿈·구분선(-----) 등 실데이터 특성 반영

---

## 2. 시스템 구성요소 (개발 대상 4개)

| # | 구성요소 | 내용 |
|---|---|---|
| ① | 데이터 적재 배치 | 메뉴별 쿼리 실행 → 클렌징 → 청크 조립 → 임베딩 → 벡터DB 저장. 재실행 시 전체 재적재(upsert) 방식 |
| ② | 벡터 DB | Qdrant Docker 설치·설정 (개발 없음) |
| ③ | 챗봇 API 서버 | FastAPI. 질문 수신 → 검색 → LLM 답변 → 출처 포함 JSON 응답 |
| ④ | 챗봇 웹 화면 | 채팅형 UI. 질문 입력, 답변 표시, 근거(데이터유형+ID+파일명) 표시 |

---

## 3. 데이터 소스: 메뉴별 조합 쿼리 (확정본)

> 공통 사항
> - 첨부는 TM_ATTFILE 공용 테이블, DOCUCD로 구분: C0002010(메뉴1 메인) / C0002011(메뉴1 이력) / C0002012(외부요청) / C0002013(심의위원회) / C0002014(내부자료)
> - 첨부는 반드시 GROUP BY + LISTAGG로 뭉쳐 조인 (행 뻥튀기 금지)
> - 파일경로(FILE_PATH)는 1단계 청크에 포함하지 않음 (2단계 첨부 파이프라인에서 별도 조회)

### 3.1 메뉴 1: 기관별 사업정보 (3단: 메인+이력+첨부×2)

```sql
SELECT A.ORGANIZATION_ID, A.CRP_NO, A.CRP_NAME, A.CRP_DEPT_NAME,
       A.BUSINESS_NAME, A.LAW_COMMENT, A.SUPPORT_COMMENT, A.CRP_GBN, A.CONTENT,
       B.DETAIL_ID, B.REQ_DTM, B.TRNS_MTD, B.STS_ISSUE_TYPE, B.STS_ISSUE_GBN,
       B.STS_ISSUE_ITEM, B.TRNS_CYCLE, B.REQ_NOTE, B.REQ_GBN, B.PAY_AMT,
       CO.ORG_ATT_CNT, CO.ORG_ATT_NAMES,
       CH.DTL_ATT_CNT, CH.DTL_ATT_NAMES
  FROM HISTORY_ORGANIZATION_INFO@STATWEB A,
       HISTORY_ORGANIZATION_DETAIL@STATWEB B,
       ( SELECT ATTFILE_ID, COUNT(*) AS ORG_ATT_CNT,
                LISTAGG(ATTFILE_NM, ', ') WITHIN GROUP (ORDER BY ATTFILE_NM) AS ORG_ATT_NAMES
           FROM TM_ATTFILE@STATWEB WHERE DOCUCD = 'C0002010' GROUP BY ATTFILE_ID ) CO,
       ( SELECT ATTFILE_ID, COUNT(*) AS DTL_ATT_CNT,
                LISTAGG(ATTFILE_NM, ', ') WITHIN GROUP (ORDER BY ATTFILE_NM) AS DTL_ATT_NAMES
           FROM TM_ATTFILE@STATWEB WHERE DOCUCD = 'C0002011' GROUP BY ATTFILE_ID ) CH
 WHERE 1=1
   AND A.ORGANIZATION_ID = B.ORGANIZATION_ID(+)
   AND A.ORGANIZATION_ID = CO.ATTFILE_ID(+)   -- ★ 메인첨부는 A에 직결 (이력 없는 기관도 첨부 유지)
   AND B.DETAIL_ID = CH.ATTFILE_ID(+)
 ORDER BY A.CR_DTM DESC, REQ_DTM DESC
```

- 관계: 기관 1 : 이력 N (이력 0건 기관 존재함 — 케이스 1·2)
- 건수: 약 836행

### 3.2 메뉴 2: 외부요청 (2단: 메인+첨부)

```sql
SELECT A.REQUEST_ID, A.CRP_NO, A.CRP_NAME, A.CRP_DEPT_NAME,
       A.REQ_PERSON_NAME, A.REQ_DTM, A.PROC_PERSON_NAME, A.TRNS_MTD,
       A.TITLE, A.REQ_MENT, A.PROC_DTM, A.PROC_MENT, A.NOTE,
       CO.ATT_CNT, CO.ATT_NAMES
  FROM HISTORY_EXTERNAL_REQUEST@STATWEB A,
       ( SELECT ATTFILE_ID, COUNT(*) AS ATT_CNT,
                LISTAGG(ATTFILE_NM, ', ') WITHIN GROUP (ORDER BY ATTFILE_NM) AS ATT_NAMES
           FROM TM_ATTFILE@STATWEB WHERE DOCUCD = 'C0002012' GROUP BY ATTFILE_ID ) CO
 WHERE A.REQUEST_ID = CO.ATTFILE_ID(+)
 ORDER BY PROC_DTM DESC
```

- 건수: 8행 (요청 1건 = 1행 = 1청크)

### 3.3 메뉴 3: 내부자료 (게시판형: 글+댓글, 댓글별 첨부)

```sql
SELECT A.REQUEST_ID, A.TITLE, A.PROC_MENT, A.USERNM, A.DECIDING_OFFICER,
       A.IN_DTM, A.CR_DTM AS MAIN_CR_DTM,
       B.IN_USERID, B.CR_DTM AS DTL_CR_DTM, B.FILE_COMMENT,
       CO.ATT_CNT, CO.ATT_NAMES
  FROM HISTORY_INTERNAL_INFO@STATWEB A,
       HISTORY_INTERNAL_DETAIL@STATWEB B,
       ( SELECT ATTFILE_ID, ATTFILE_NUM, COUNT(*) AS ATT_CNT,
                LISTAGG(ATTFILE_NM, ', ') WITHIN GROUP (ORDER BY ATTFILE_NM) AS ATT_NAMES
           FROM TM_ATTFILE@STATWEB WHERE DOCUCD = 'C0002014'
          GROUP BY ATTFILE_ID, ATTFILE_NUM ) CO
 WHERE 1=1
   AND A.REQUEST_ID = B.REQUEST_ID(+)
   AND B.REQUEST_ID = CO.ATTFILE_ID(+)
   AND B.ATTFILE_NUM = CO.ATTFILE_NUM(+)
 ORDER BY A.CR_DTM DESC, B.CR_DTM DESC
```

- 관계: 글 1 : 댓글 N, 첨부는 댓글 단위(2중 키: REQUEST_ID + ATTFILE_NUM)
- 건수: 36행 = 글 13건 + 댓글. **청크는 글 단위 13개 (스크립트 그룹핑 조립)**
- ※ 원 쿼리의 CR_DTM 중복을 별칭(MAIN_CR_DTM / DTL_CR_DTM)으로 분리함 — 적재 스크립트는 이 별칭 기준

### 3.4 메뉴 4: 심의위원회 (2단: 메인+첨부)

```sql
SELECT A.DISCUSSION_ID, A.CRP_NO, A.CRP_NAME, A.CRP_DEPT_NAME,
       A.TITLE, A.REQ_MENT, A.LAW_COMMENT, A.NOTE,
       A.AGENDA_NUMBER, A.YEAR, A.SEQ,
       A.DISCUSSION_RESULT, A.RESULT_CONTENT, A.REQ_ITEM, A.PROC_ITEM,
       CO.ATT_CNT, CO.ATT_NAMES
  FROM HISTORY_DISCUSSION_COMMITTEE@STATWEB A,
       ( SELECT ATTFILE_ID, COUNT(*) AS ATT_CNT,
                LISTAGG(ATTFILE_NM, ', ') WITHIN GROUP (ORDER BY ATTFILE_NM) AS ATT_NAMES
           FROM TM_ATTFILE@STATWEB WHERE DOCUCD = 'C0002013' GROUP BY ATTFILE_ID ) CO
 WHERE A.DISCUSSION_ID = CO.ATTFILE_ID(+)
 ORDER BY YEAR DESC, SEQ DESC, AGENDA_NUMBER DESC
```

- 건수: 57행 (안건 1건 = 1청크). 첨부 최대 7개 (LISTAGG 안전)

---

## 4. 컬럼 분류표 (본문/메타 배치 확정)

구분 범례: ①서술형(본문 핵심) ②코드성(해당 없음 — 전 메뉴 한글값) ③필터·식별 ④민감

### 메뉴 1: 기관별 사업정보

| 컬럼 | 구분 | 본문 | 메타 | 비고 |
|---|---|---|---|---|
| ORGANIZATION_ID | ③ | ✕ | ✅ | 기관 키 |
| CRP_NO | ④ | ✕ | ✅ | 사업자번호 — 본문 제외 확정 |
| CRP_NAME | ③ | ✅ | ✅ | 기관명 |
| CRP_DEPT_NAME | ③ | ✅ | ✕ | |
| BUSINESS_NAME | ① | ✅ | ✕ | |
| LAW_COMMENT | ① | ✅ | ✕ | |
| SUPPORT_COMMENT | ① | ✅ | ✕ | |
| CRP_GBN | ③ | ✅ | ✅ | 기관분류 |
| CONTENT | ① | ✅ | ✕ | |
| DETAIL_ID | ③ | ✅(라벨) | ✅ | |
| REQ_DTM | ③ | ✅ | ✅(연도) | |
| TRNS_MTD | ③ | ✅ | ✕ | |
| STS_ISSUE_TYPE | ③ | ✅ | ✅ | 통계종류 |
| STS_ISSUE_GBN | ③ | ✅ | ✅ | 수출입구분 |
| STS_ISSUE_ITEM | ① | ✅ | ✕ | |
| TRNS_CYCLE | ③ | ✅ | ✕ | |
| REQ_NOTE | ①★ | ✅ | ✕ | 500자 핵심 |
| REQ_GBN | ③ | ✅ | ✕ | |
| PAY_AMT | ④→포함 | ✅ | ✕ | 수수료 — 본문 포함 확정 |
| 첨부 4컬럼 | 표시 | ✅ | ✕ | |

### 메뉴 2: 외부요청

| 컬럼 | 본문 | 메타 | 비고 |
|---|---|---|---|
| REQUEST_ID | ✅(라벨) | ✅ | |
| CRP_NO | ✕ | ✅ | |
| CRP_NAME | ✅ | ✅ | |
| CRP_DEPT_NAME | ✅ | ✕ | |
| REQ_PERSON_NAME | ✅ | ✅ | 요청자 — 포함 확정 |
| REQ_DTM | ✅ | ✅(연도) | |
| PROC_PERSON_NAME | ✅ | ✅ | 제공자 — 포함 확정 |
| TRNS_MTD | ✅ | ✕ | |
| TITLE / REQ_MENT / PROC_MENT / NOTE | ✅ | ✕ | 서술형 |
| PROC_DTM | ✅ | ✕ | |
| 첨부 2컬럼 | ✅ | ✕ | |

### 메뉴 3: 내부자료

| 컬럼 | 본문 | 메타 | 비고 |
|---|---|---|---|
| REQUEST_ID | ✅(라벨) | ✅ | |
| TITLE | ✅ | ✕ | |
| PROC_MENT | ✅ | ✕ | 500자 핵심 |
| USERNM | ✅ | ✅ | 담당자 |
| DECIDING_OFFICER | ✅ | ✅ | 결재자 — 포함 확정 |
| MAIN_CR_DTM | ✅ | ✅(연도) | |
| IN_USERID | 보류 | ✕ | 값이 이름이면 댓글 줄에 포함, ID면 제외 (확인 대기) |
| DTL_CR_DTM | ✅(댓글) | ✕ | |
| FILE_COMMENT | ✅(댓글) | ✕ | |
| 첨부 2컬럼 | ✅(댓글별) | ✕ | |

### 메뉴 4: 심의위원회

| 컬럼 | 본문 | 메타 | 비고 |
|---|---|---|---|
| DISCUSSION_ID | ✅(라벨) | ✅ | |
| CRP_NO | ✕ | ✅ | |
| CRP_NAME | ✅ | ✅ | |
| CRP_DEPT_NAME | ✅ | ✕ | |
| TITLE / REQ_MENT / LAW_COMMENT / RESULT_CONTENT / REQ_ITEM / PROC_ITEM | ✅ | ✕ | 서술형 5개+ |
| NOTE | ✅(있을 때만) | ✕ | 빈값 많음 |
| AGENDA_NUMBER | ✅ | ✕ | |
| YEAR | ✅ | ✅ | 연도 필터 핵심 |
| SEQ | ✅ | ✅ | 차수 |
| DISCUSSION_RESULT | ✅ | ✅(정규화) | 원안수용/제공불가/항목조정후 제공 등 — 공백 제거 정규화 후 메타 저장 |
| 첨부 2컬럼 | ✅ | ✕ | |

---

## 5. 청크 조립 규칙 및 템플릿

### 5.0 공통 규칙

1. **클렌징 함수를 모든 서술형 값에 적용 후 조립:**
   - HTML 태그 제거, 장식 기호(■★ / ----- 등) 제거, 연속 줄바꿈은 2개로, 연속 공백은 1개로, 앞뒤 공백 제거
2. **NULL 처리:** 값이 없는 항목은 라인 자체를 생략 (빈 라벨 출력 금지). 첨부 없으면 "첨부문서 없음" 대신 라인 생략
3. **메타데이터 공통 필드:** `데이터유형`, `원본ID`, `기관명`(해당 시), `연도`(해당 시)
4. FILE_PATH·경로 정보는 본문·메타 어디에도 넣지 않음 (2단계에서 처리)
5. 임베딩 입력 = 본문 텍스트 전체. 저장 시 (벡터, 본문, 메타) 세트로 upsert

### 5.1 메뉴 1 템플릿 — 이력 1건 = 1청크 / 기관 소개 청크 분기

**분기 규칙: DETAIL_ID가 NULL이면 [기관정보] 청크, 있으면 [신청이력] 청크**

[신청이력] 청크 (DETAIL_ID 있음):
```
[신청이력 {DETAIL_ID}] {CRP_NAME}({CRP_GBN}, {CRP_DEPT_NAME})이(가) {REQ_DTM} 신청.
사업명: {BUSINESS_NAME}
제공목적: {SUPPORT_COMMENT}
요청구분: {REQ_GBN} / 통계종류: {STS_ISSUE_TYPE} / 수출입구분: {STS_ISSUE_GBN}
제공항목: {STS_ISSUE_ITEM}
전송방식: {TRNS_MTD} / 제공주기: {TRNS_CYCLE} / 수수료: {PAY_AMT}
요청내용: {REQ_NOTE}
기관자료: {ORG_ATT_NAMES}
첨부문서: {DTL_ATT_NAMES}
```
메타: {데이터유형:"신청이력", 원본ID:DETAIL_ID, 기관ID:ORGANIZATION_ID, 사업자번호:CRP_NO, 기관명:CRP_NAME, 기관분류:CRP_GBN, 연도:REQ_DTM[:4], 통계종류:STS_ISSUE_TYPE, 수출입구분:STS_ISSUE_GBN}

[기관정보] 청크 (DETAIL_ID NULL — 이력 없는 기관):
```
[기관정보] {CRP_NAME}({CRP_GBN}, {CRP_DEPT_NAME})
사업명: {BUSINESS_NAME}
법적근거: {LAW_COMMENT}
제공목적: {SUPPORT_COMMENT}
주요내용: {CONTENT}
기관자료: {ORG_ATT_NAMES}
```
메타: {데이터유형:"기관정보", 원본ID:ORGANIZATION_ID, 사업자번호:CRP_NO, 기관명:CRP_NAME, 기관분류:CRP_GBN}

※ 참고: 이력 있는 기관도 기관정보 청크를 별도 1개 생성함 (전 기관 프로필 확보). 신청이력 청크의 법적근거·주요내용 중복을 피하기 위해 신청이력 템플릿에는 LAW_COMMENT·CONTENT를 넣지 않고 기관정보 청크에만 배치함.

### 5.2 메뉴 2 템플릿 — 요청 1건 = 1청크

```
[외부요청 {REQUEST_ID}] {CRP_NAME}({CRP_DEPT_NAME})의 {REQ_PERSON_NAME}이(가) {REQ_DTM} 요청.
제목: {TITLE}
요청내용: {REQ_MENT}
처리내용: {PROC_MENT} (제공자: {PROC_PERSON_NAME}, 제공일: {PROC_DTM}, 제공방식: {TRNS_MTD})
특이사항: {NOTE}
첨부문서: {ATT_NAMES}
```
메타: {데이터유형:"외부요청", 원본ID:REQUEST_ID, 사업자번호:CRP_NO, 기관명:CRP_NAME, 요청자:REQ_PERSON_NAME, 제공자:PROC_PERSON_NAME, 연도:REQ_DTM[:4]}

### 5.3 메뉴 3 템플릿 — 글 1건 = 1청크 (REQUEST_ID 그룹핑 조립)

조립 로직: 쿼리 결과를 REQUEST_ID로 그룹핑. 첫 행에서 A 컬럼을, 각 행에서 B 컬럼(댓글)을 취해 시간순 나열.

```
[내부자료 {REQUEST_ID}] {TITLE} (담당: {USERNM}, 결재: {DECIDING_OFFICER}, 등록: {MAIN_CR_DTM})
주요내용: {PROC_MENT}

댓글:
- {IN_USERID*} ({DTL_CR_DTM}): {FILE_COMMENT} [첨부: {ATT_NAMES}]
- ... (댓글 수만큼 반복, 댓글 없으면 "댓글:" 섹션 생략)
```
메타: {데이터유형:"내부자료", 원본ID:REQUEST_ID, 담당자:USERNM, 결재자:DECIDING_OFFICER, 연도:MAIN_CR_DTM[:4]}

※ IN_USERID*: 값이 실명이면 표기, 사번/ID 형식이면 생략 (확인 대기 — 기본값: 생략)

### 5.4 메뉴 4 템플릿 — 안건 1건 = 1청크

```
[심의안건 {DISCUSSION_ID}] {YEAR}년 {SEQ}차 위원회 안건 {AGENDA_NUMBER} — {CRP_NAME}({CRP_DEPT_NAME})
제목: {TITLE}
요청내용: {REQ_MENT}
법적근거: {LAW_COMMENT}
요청항목: {REQ_ITEM}
심의결과: {DISCUSSION_RESULT}
결정내용: {RESULT_CONTENT}
제공항목: {PROC_ITEM}
특이사항: {NOTE}
첨부문서: {ATT_NAMES}
```
메타: {데이터유형:"심의위원회", 원본ID:DISCUSSION_ID, 사업자번호:CRP_NO, 기관명:CRP_NAME, 연도:YEAR, 차수:SEQ, 심의결과:normalize(DISCUSSION_RESULT)}

---

## 6. 챗봇 API 서버 요구사항

1. `POST /chat` — 질문 수신 → 질문 임베딩 → 벡터 검색(top 5~8) → LLM 답변 생성 → 응답
2. 검색 시 메타데이터 필터 지원: 데이터유형, 기관명, 연도, 통계종류, 심의결과 등 (질문에서 필터 조건 추출은 LLM 활용 가능)
3. LLM 프롬프트 원칙:
   - 검색된 청크만 근거로 답변, 근거 없으면 "관련 이력을 찾지 못했습니다"
   - 자료의 용어·표현을 그대로 사용, 새 단어 생성 금지
   - temperature 0.2 이하
4. 응답에 출처 필수 포함: [{데이터유형, 원본ID, 기관명}] 목록
5. 대화 이력은 세션 단위 유지 (후속 질문 대응)

## 7. 검증 기준 (필수 테스트)

1. 적재 건수 대조: 메뉴1 = HISTORY_ORGANIZATION_DETAIL 건수 + 이력 없는 기관 수(기관정보 청크는 전 기관 수) / 메뉴2 = 8 / 메뉴3 = 글 수 13 / 메뉴4 = 57 (더미 기준은 생성기 수치와 대조)
2. 4가지 케이스(①메인만 ②메인+메인첨부 ③이력+이력첨부 ④전부) 각각 청크 정상 생성 확인
3. 첨부 다건(2개 이상) 시 행 중복 없이 파일명이 한 칸에 뭉치는지 확인
4. 메뉴3 그룹핑: 댓글 여러 개인 글이 1청크로 합쳐지는지 확인
5. 검색 시나리오 테스트: 기관명 검색 / 연도 필터 / 심의결과 필터 / 담당자·요청자 검색 / 존재하지 않는 건 질문 시 "없음" 응답

## 8. 보류·확인 항목

| 항목 | 상태 |
|---|---|
| 메뉴3 IN_USERID 값 형식 (실명/ID) | 확인 대기 — 기본 생략으로 개발 |
| 메뉴1 REQ_NOTE 장문 건 분할 여부 | PoC에서는 1건=1청크 유지, 검색 품질 튜닝 시 재검토 |
| 개인 성명(요청자·제공자·담당자·결재자)의 외부 API 전송 | PoC 계획서에 데이터 범위로 명시 필요 |
| 하이브리드 검색(BM25+벡터) 도입 | PoC는 벡터 단독, 품질 미달 시 추가 |
