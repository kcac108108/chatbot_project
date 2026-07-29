-- ============================================================
-- 더미 데이터 생성기 (정의서 1.3절 기준)
-- 대상: CHATBOT 스키마 (로컬 개발 전용, 실데이터 절대 아님)
-- 실행: sqlplus CHATBOT/Chatbot1234@//localhost:1521/xepdb1 @generate_dummy.sql
--
-- 메뉴1(기관별 사업정보) 4가지 케이스:
--   ORG0000001~0025 : ①메인만
--   ORG0000026~0050 : ②메인+메인첨부
--   ORG0000051~0075 : ③메인+이력+이력첨부
--   ORG0000076~0100 : ④전부(메인첨부+이력+이력첨부)
-- ============================================================

SET DEFINE OFF
SET SERVEROUTPUT ON SIZE UNLIMITED
WHENEVER SQLERROR CONTINUE

PROMPT ==== 0. 기존 더미 데이터 초기화 ====
DELETE FROM TM_ATTFILE;
DELETE FROM HISTORY_ORGANIZATION_DETAIL;
DELETE FROM HISTORY_ORGANIZATION_INFO;
DELETE FROM HISTORY_EXTERNAL_REQUEST;
DELETE FROM HISTORY_INTERNAL_DETAIL;
DELETE FROM HISTORY_INTERNAL_INFO;
DELETE FROM HISTORY_DISCUSSION_COMMITTEE;
COMMIT;

PROMPT ==== 1. 더미 생성 헬퍼 패키지 생성 ====
CREATE OR REPLACE PACKAGE pkg_dummy_gen AS
  FUNCTION rand_int(p_min NUMBER, p_max NUMBER) RETURN NUMBER;
  FUNCTION rand_pick(p_list SYS.ODCIVARCHAR2LIST) RETURN VARCHAR2;
  FUNCTION rand_date(p_days_back NUMBER, p_days_forward NUMBER DEFAULT 0) RETURN DATE;
  FUNCTION rand_yyyymmdd(p_days_back NUMBER, p_days_forward NUMBER DEFAULT 0) RETURN VARCHAR2;
  FUNCTION gen_paragraph(p_min_len NUMBER, p_max_len NUMBER) RETURN VARCHAR2;
  FUNCTION gen_short_phrase(p_max_len NUMBER) RETURN VARCHAR2;
  FUNCTION gen_person_name RETURN VARCHAR2;
  FUNCTION gen_org_name RETURN VARCHAR2;
  FUNCTION gen_dept_name RETURN VARCHAR2;
  FUNCTION gen_filename(p_ext OUT VARCHAR2) RETURN VARCHAR2;
  FUNCTION gen_crp_no RETURN VARCHAR2;
  FUNCTION gen_tel RETURN VARCHAR2;
END pkg_dummy_gen;
/

CREATE OR REPLACE PACKAGE BODY pkg_dummy_gen AS

  FUNCTION rand_int(p_min NUMBER, p_max NUMBER) RETURN NUMBER IS
  BEGIN
    RETURN TRUNC(DBMS_RANDOM.VALUE(p_min, p_max + 1));
  END;

  FUNCTION rand_pick(p_list SYS.ODCIVARCHAR2LIST) RETURN VARCHAR2 IS
  BEGIN
    RETURN p_list(rand_int(1, p_list.COUNT));
  END;

  FUNCTION rand_date(p_days_back NUMBER, p_days_forward NUMBER DEFAULT 0) RETURN DATE IS
  BEGIN
    RETURN TRUNC(SYSDATE) - rand_int(0, p_days_back) + rand_int(0, p_days_forward);
  END;

  FUNCTION rand_yyyymmdd(p_days_back NUMBER, p_days_forward NUMBER DEFAULT 0) RETURN VARCHAR2 IS
  BEGIN
    RETURN TO_CHAR(rand_date(p_days_back, p_days_forward), 'YYYYMMDD');
  END;

  FUNCTION gen_person_name RETURN VARCHAR2 IS
    v_surname SYS.ODCIVARCHAR2LIST := SYS.ODCIVARCHAR2LIST('김','이','박','최','정','강','조','윤','장','임');
    v_given   SYS.ODCIVARCHAR2LIST := SYS.ODCIVARCHAR2LIST('민준','서연','지훈','수빈','다은','현우','예진','도윤','하은','준서','태호','유진');
  BEGIN
    RETURN rand_pick(v_surname) || rand_pick(v_given);
  END;

  FUNCTION gen_org_name RETURN VARCHAR2 IS
    v_prefix SYS.ODCIVARCHAR2LIST := SYS.ODCIVARCHAR2LIST('대한','한국','동양','글로벌','신성','미래','중앙','대성','경남','서울','태평양','우성');
    v_core   SYS.ODCIVARCHAR2LIST := SYS.ODCIVARCHAR2LIST('산업','무역','화학','전자','철강','물류','식품','에너지','바이오','정밀','기계','섬유');
    v_suffix SYS.ODCIVARCHAR2LIST := SYS.ODCIVARCHAR2LIST('(주)','협회','연구원','공사','센터','재단');
  BEGIN
    RETURN rand_pick(v_prefix) || rand_pick(v_core) || rand_pick(v_suffix);
  END;

  FUNCTION gen_dept_name RETURN VARCHAR2 IS
    v_dept SYS.ODCIVARCHAR2LIST := SYS.ODCIVARCHAR2LIST('기획조정실','수출입관리팀','경영지원팀','통계분석팀','대외협력팀','정보전산팀','법무팀','감사실');
  BEGIN
    RETURN rand_pick(v_dept);
  END;

  FUNCTION gen_crp_no RETURN VARCHAR2 IS
  BEGIN
    RETURN LPAD(rand_int(100,999),3,'0')||'-'||LPAD(rand_int(10,99),2,'0')||'-'||LPAD(rand_int(10000,99999),5,'0');
  END;

  FUNCTION gen_tel RETURN VARCHAR2 IS
  BEGIN
    RETURN '02-'||LPAD(rand_int(100,999),3,'0')||'-'||LPAD(rand_int(1000,9999),4,'0');
  END;

  -- p_min_len/p_max_len 단위는 BYTE (컬럼이 VARCHAR2(n BYTE)이므로).
  FUNCTION gen_short_phrase(p_max_len NUMBER) RETURN VARCHAR2 IS
    v_word SYS.ODCIVARCHAR2LIST := SYS.ODCIVARCHAR2LIST(
      '수출입','통계','산업동향','기업지원','정책수립','자료제공','품목별','국가별','월별','분기별',
      '실적분석','무역수지','생산현황','고용동향','물가지수','에너지','바이오','정밀기계','섬유산업',
      '전자부품','화학원료','철강','물류','재무현황','경영지원','법령검토','개인정보','보호조치',
      '승인절차','신청내역','지원사업','현황조사','자료열람','품질관리'
    );
    v_result VARCHAR2(200) := rand_pick(v_word);
    v_n NUMBER := rand_int(1,2);
  BEGIN
    FOR k IN 1..v_n LOOP
      v_result := v_result || ' ' || rand_pick(v_word);
    END LOOP;
    WHILE LENGTHB(v_result) > p_max_len LOOP
      v_result := SUBSTR(v_result, 1, LENGTH(v_result)-1);
    END LOOP;
    RETURN v_result;
  END;

  -- 서술형 문단 생성: 한글 문장 조각을 이어붙이고, 실데이터 특성(장식기호/구분선/html태그/공백노이즈)을 확률적으로 반영
  -- p_min_len/p_max_len 단위는 BYTE
  FUNCTION gen_paragraph(p_min_len NUMBER, p_max_len NUMBER) RETURN VARCHAR2 IS
    v_frag SYS.ODCIVARCHAR2LIST := SYS.ODCIVARCHAR2LIST(
      '본 자료는 관련 법령에 근거하여 제출된 통계자료이다.',
      '신청 기관은 최근 3년간 수출입 실적을 기준으로 자료를 요청하였다.',
      '제공된 통계는 품목별 무역수지 분석에 활용될 예정이다.',
      '해당 기관은 정기적으로 통계 자료를 제공받아 내부 보고서에 반영하고 있다.',
      '자료 제공 목적은 산업 동향 파악 및 정책 수립 지원이다.',
      '요청 내용에는 특정 품목의 월별 수출입 물량이 포함되어 있다.',
      '심의위원회는 개인정보 보호 및 통계법 준수 여부를 검토하였다.',
      '제공 항목에는 기업명, 품목코드, 거래금액 등이 포함된다.',
      '본 건은 이전 신청 이력과 연계하여 검토가 필요하다.',
      '담당 부서는 자료의 활용 목적을 재확인한 후 승인 절차를 진행하였다.',
      '제공 방식은 암호화된 전자파일 형태로 전송되었다.',
      '해당 통계는 비공개 원자료를 가공하여 작성된 것이다.',
      '요청 기관의 사업 목적과 통계 활용 계획이 명확히 기재되어 있다.',
      '자료 제공 이후 활용 현황에 대한 사후 점검이 이루어질 예정이다.',
      '관련 법령 제12조에 따라 자료 제공 범위가 제한된다.',
      '본 신청 건은 유사 사례와 비교하여 검토되었다.',
      '제공되는 통계는 분기별로 갱신되어 재전송된다.',
      '심의 결과에 따라 일부 항목은 제공이 제한되었다.',
      '해당 기업은 관련 기관에 등록된 기업으로 확인되었다.',
      '자료 활용 목적이 상업적 이용에 해당하지 않음을 확인하였다.',
      '제공 이력은 향후 감사 자료로 활용될 수 있다.',
      '신청서에는 자료의 구체적인 활용 범위가 명시되어 있다.',
      '담당자는 유선 및 이메일을 통해 추가 확인 절차를 진행하였다.',
      '본 건은 정기 심의 안건으로 상정되어 논의되었다.'
    );
    v_result VARCHAR2(4000) := NULL;
    v_sep VARCHAR2(20);
    v_roll NUMBER;
  BEGIN
    WHILE v_result IS NULL OR LENGTHB(v_result) < p_min_len LOOP
      v_roll := rand_int(1,100);
      IF v_roll <= 10 THEN
        v_sep := CHR(10)||'-----'||CHR(10);
      ELSIF v_roll <= 25 THEN
        v_sep := CHR(10);
      ELSIF v_roll <= 35 THEN
        v_sep := ' <br> ';
      ELSE
        v_sep := ' ';
      END IF;
      v_result := v_result || CASE WHEN v_result IS NULL THEN '' ELSE v_sep END || rand_pick(v_frag);
    END LOOP;

    IF rand_int(1,100) <= 20 THEN
      v_result := rand_pick(SYS.ODCIVARCHAR2LIST('■ ','★ ','▶ ')) || v_result;
    END IF;
    IF rand_int(1,100) <= 15 THEN
      v_result := '  ' || v_result || CHR(10) || CHR(10);
    END IF;

    -- 바이트 기준 트림 (SUBSTRB는 멀티바이트 문자를 깨뜨릴 수 있어 문자 단위로 한 글자씩 줄임)
    WHILE LENGTHB(v_result) > p_max_len LOOP
      v_result := SUBSTR(v_result, 1, LENGTH(v_result)-1);
    END LOOP;
    RETURN v_result;
  END;

  FUNCTION gen_filename(p_ext OUT VARCHAR2) RETURN VARCHAR2 IS
    v_topic SYS.ODCIVARCHAR2LIST := SYS.ODCIVARCHAR2LIST('사업계획서','신청서','통계자료','회의록','검토보고서','첨부자료','계약서','증빙서류');
    v_ext   SYS.ODCIVARCHAR2LIST := SYS.ODCIVARCHAR2LIST('pdf','hwp','xlsx','docx','zip');
    v_year  NUMBER := rand_int(2021,2026);
  BEGIN
    p_ext := rand_pick(v_ext);
    RETURN v_year||'년_'||rand_pick(v_topic)||'_'||rand_int(1,99)||'.'||p_ext;
  END;

END pkg_dummy_gen;
/

SHOW ERRORS

PROMPT ==== 2. 메뉴1: 기관정보(HISTORY_ORGANIZATION_INFO) 100건 ====
DECLARE
  v_org_id  VARCHAR2(15);
  v_crp_gbn VARCHAR2(100);
BEGIN
  FOR i IN 1..100 LOOP
    v_org_id  := 'ORG'||LPAD(i,7,'0');
    v_crp_gbn := pkg_dummy_gen.rand_pick(SYS.ODCIVARCHAR2LIST('공공기관','민간기업','연구기관','협회단체','학교법인'));
    INSERT INTO HISTORY_ORGANIZATION_INFO (
      ORGANIZATION_ID, CRP_NO, CRP_NAME, CRP_DEPT_NAME, BUSINESS_NAME,
      LAW_COMMENT, SUPPORT_COMMENT, CRP_GBN, CONTENT, IN_DTM, IN_USERID, CR_DTM, CR_USERID
    ) VALUES (
      v_org_id, pkg_dummy_gen.gen_crp_no(), pkg_dummy_gen.gen_org_name(),
      CASE WHEN pkg_dummy_gen.rand_int(1,100) <= 85 THEN pkg_dummy_gen.gen_dept_name() ELSE NULL END,
      pkg_dummy_gen.gen_short_phrase(90),
      CASE WHEN pkg_dummy_gen.rand_int(1,100) <= 90 THEN pkg_dummy_gen.gen_paragraph(60,280) ELSE NULL END,
      pkg_dummy_gen.gen_paragraph(150,900),
      v_crp_gbn,
      pkg_dummy_gen.gen_paragraph(300,1500),
      pkg_dummy_gen.rand_date(1800), 'sysuser01',
      pkg_dummy_gen.rand_date(1800), 'sysuser01'
    );
  END LOOP;
  COMMIT;
END;
/

PROMPT ==== 3. 메뉴1: 메인첨부(TM_ATTFILE, C0002010) — 케이스②(026~050) + 케이스④(076~100) ====
DECLARE
  v_org_id VARCHAR2(15);
  v_cnt    NUMBER;
  v_ext    VARCHAR2(20);
  v_name   VARCHAR2(150);
BEGIN
  FOR i IN 26..100 LOOP
    IF i <= 50 OR i >= 76 THEN
      v_org_id := 'ORG'||LPAD(i,7,'0');
      v_cnt := pkg_dummy_gen.rand_int(1,7);
      FOR n IN 1..v_cnt LOOP
        v_name := pkg_dummy_gen.gen_filename(v_ext);
        INSERT INTO TM_ATTFILE (
          DOCUCD, ATTFILE_ID, ATTFILE_NUM, ATTFILE_NM, FILE_PATH, FILE_EXT,
          FILE_SIZE, DOWNLOAD_CNT, CR_USERID, CR_DTM, IN_USERID, IN_DTM
        ) VALUES (
          'C0002010', v_org_id, n, v_name, '/data/dummy/attach/'||v_org_id||'_'||n||'.'||v_ext, v_ext,
          pkg_dummy_gen.rand_int(10000,5000000), pkg_dummy_gen.rand_int(0,500),
          'sysuser01', pkg_dummy_gen.rand_date(1800), 'sysuser01', pkg_dummy_gen.rand_date(900)
        );
      END LOOP;
    END IF;
  END LOOP;
  COMMIT;
END;
/

PROMPT ==== 4. 메뉴1: 이력(HISTORY_ORGANIZATION_DETAIL) + 이력첨부(C0002011) — 케이스③(051~075) + 케이스④(076~100) ====
DECLARE
  v_org_id  VARCHAR2(15);
  v_det_id  VARCHAR2(15);
  v_det_seq NUMBER := 0;
  v_hist_cnt NUMBER;
  v_att_cnt  NUMBER;
  v_ext      VARCHAR2(20);
  v_name     VARCHAR2(150);
BEGIN
  FOR i IN 51..100 LOOP
    v_org_id := 'ORG'||LPAD(i,7,'0');
    v_hist_cnt := pkg_dummy_gen.rand_int(1,4);
    FOR h IN 1..v_hist_cnt LOOP
      v_det_seq := v_det_seq + 1;
      v_det_id := 'DET'||LPAD(v_det_seq,7,'0');
      INSERT INTO HISTORY_ORGANIZATION_DETAIL (
        ORGANIZATION_ID, DETAIL_ID, REQ_DTM, TRNS_MTD, STS_ISSUE_TYPE, STS_ISSUE_GBN,
        STS_ISSUE_ITEM, TRNS_CYCLE, REQ_NOTE, REQ_GBN, IN_DTM, IN_USERID, CR_DTM, CR_USERID, PAY_AMT
      ) VALUES (
        v_org_id, v_det_id, pkg_dummy_gen.rand_yyyymmdd(1800),
        pkg_dummy_gen.rand_pick(SYS.ODCIVARCHAR2LIST('이메일','우편','직접방문','FTP','전용회선')),
        pkg_dummy_gen.rand_pick(SYS.ODCIVARCHAR2LIST('수출입통계','생산통계','물가통계','고용통계','무역통계')),
        CASE WHEN pkg_dummy_gen.rand_int(1,100) <= 70 THEN pkg_dummy_gen.rand_pick(SYS.ODCIVARCHAR2LIST('수출','수입','수출입')) ELSE NULL END,
        pkg_dummy_gen.gen_paragraph(60,180),
        pkg_dummy_gen.rand_pick(SYS.ODCIVARCHAR2LIST('월간','분기','반기','연간','수시')),
        pkg_dummy_gen.gen_paragraph(300,1400),
        pkg_dummy_gen.rand_pick(SYS.ODCIVARCHAR2LIST('신규','갱신','변경','해지')),
        pkg_dummy_gen.rand_date(1800), 'sysuser01', pkg_dummy_gen.rand_date(1800), 'sysuser01',
        CASE WHEN pkg_dummy_gen.rand_int(1,100) <= 80 THEN TO_CHAR(pkg_dummy_gen.rand_int(1,50)*10000) ELSE NULL END
      );
      v_att_cnt := pkg_dummy_gen.rand_int(0,7);
      FOR n IN 1..v_att_cnt LOOP
        v_name := pkg_dummy_gen.gen_filename(v_ext);
        INSERT INTO TM_ATTFILE (
          DOCUCD, ATTFILE_ID, ATTFILE_NUM, ATTFILE_NM, FILE_PATH, FILE_EXT,
          FILE_SIZE, DOWNLOAD_CNT, CR_USERID, CR_DTM, IN_USERID, IN_DTM
        ) VALUES (
          'C0002011', v_det_id, n, v_name, '/data/dummy/attach/'||v_det_id||'_'||n||'.'||v_ext, v_ext,
          pkg_dummy_gen.rand_int(10000,5000000), pkg_dummy_gen.rand_int(0,500),
          'sysuser01', pkg_dummy_gen.rand_date(1800), 'sysuser01', pkg_dummy_gen.rand_date(900)
        );
      END LOOP;
    END LOOP;
  END LOOP;
  COMMIT;
END;
/

PROMPT ==== 5. 메뉴2: 외부요청(HISTORY_EXTERNAL_REQUEST) 8건 + 첨부(C0002012) ====
DECLARE
  v_req_id VARCHAR2(15);
  v_cnt    NUMBER;
  v_ext    VARCHAR2(20);
  v_name   VARCHAR2(150);
  v_processed NUMBER;
BEGIN
  FOR i IN 1..8 LOOP
    v_req_id := 'EXT'||LPAD(i,7,'0');
    v_processed := CASE WHEN i <= 6 THEN 1 ELSE 0 END; -- 8건 중 2건은 아직 미처리(NULL 케이스)
    INSERT INTO HISTORY_EXTERNAL_REQUEST (
      REQUEST_ID, CRP_NO, CRP_NAME, CRP_DEPT_NAME, REQ_PERSON_NAME, REQ_DTM, REQ_TEL,
      PROC_PERSON_NAME, TRNS_MTD, TITLE, REQ_MENT, PROC_DTM, PROC_MENT, NOTE,
      IN_DTM, IN_USERID, CR_DTM, CR_USERID
    ) VALUES (
      v_req_id, pkg_dummy_gen.gen_crp_no(), pkg_dummy_gen.gen_org_name(), pkg_dummy_gen.gen_dept_name(),
      pkg_dummy_gen.gen_person_name(), pkg_dummy_gen.rand_yyyymmdd(1800), pkg_dummy_gen.gen_tel(),
      CASE WHEN v_processed = 1 THEN pkg_dummy_gen.gen_person_name() ELSE NULL END,
      pkg_dummy_gen.rand_pick(SYS.ODCIVARCHAR2LIST('이메일','우편','직접방문','FTP')),
      '외부요청 '||pkg_dummy_gen.rand_pick(SYS.ODCIVARCHAR2LIST('통계자료 제공 요청','수출입 실적 조회 요청','자료 열람 요청','통계 발췌 요청')),
      pkg_dummy_gen.gen_paragraph(300,1900),
      CASE WHEN v_processed = 1 THEN pkg_dummy_gen.rand_yyyymmdd(1700) ELSE NULL END,
      CASE WHEN v_processed = 1 THEN pkg_dummy_gen.gen_paragraph(200,1400) ELSE NULL END,
      CASE WHEN pkg_dummy_gen.rand_int(1,100) <= 50 THEN pkg_dummy_gen.gen_paragraph(40,450) ELSE NULL END,
      pkg_dummy_gen.rand_date(1800), 'sysuser01', pkg_dummy_gen.rand_date(1800), 'sysuser01'
    );
    v_cnt := pkg_dummy_gen.rand_int(0,5);
    FOR n IN 1..v_cnt LOOP
      v_name := pkg_dummy_gen.gen_filename(v_ext);
      INSERT INTO TM_ATTFILE (
        DOCUCD, ATTFILE_ID, ATTFILE_NUM, ATTFILE_NM, FILE_PATH, FILE_EXT,
        FILE_SIZE, DOWNLOAD_CNT, CR_USERID, CR_DTM, IN_USERID, IN_DTM
      ) VALUES (
        'C0002012', v_req_id, n, v_name, '/data/dummy/attach/'||v_req_id||'_'||n||'.'||v_ext, v_ext,
        pkg_dummy_gen.rand_int(10000,5000000), pkg_dummy_gen.rand_int(0,500),
        'sysuser01', pkg_dummy_gen.rand_date(1800), 'sysuser01', pkg_dummy_gen.rand_date(900)
      );
    END LOOP;
  END LOOP;
  COMMIT;
END;
/

PROMPT ==== 6. 메뉴3: 내부자료(HISTORY_INTERNAL_INFO) 13건 + 댓글(HISTORY_INTERNAL_DETAIL) 23건 + 첨부(C0002014) ====
DECLARE
  TYPE t_num_list IS TABLE OF NUMBER;
  v_comment_counts t_num_list := t_num_list(3,2,1,0,4,2,1,3,0,2,2,2,1); -- 합계 23 (13건)
  v_req_id VARCHAR2(15);
  v_has_file NUMBER;
  v_ext VARCHAR2(20);
  v_name VARCHAR2(150);
BEGIN
  FOR i IN 1..13 LOOP
    v_req_id := 'INT'||LPAD(i,7,'0');
    INSERT INTO HISTORY_INTERNAL_INFO (
      REQUEST_ID, TITLE, PROC_MENT, USERNM, DECIDING_OFFICER, IN_DTM, IN_USERID, CR_DTM, CR_USERID
    ) VALUES (
      v_req_id, '내부자료 '||pkg_dummy_gen.rand_pick(SYS.ODCIVARCHAR2LIST('통계자료 제공 검토','신규 신청 건 처리','자료 제공 범위 조정','정기 점검 결과')),
      pkg_dummy_gen.gen_paragraph(300,1900),
      pkg_dummy_gen.gen_person_name(),
      CASE WHEN pkg_dummy_gen.rand_int(1,100) <= 90 THEN pkg_dummy_gen.gen_person_name() ELSE NULL END,
      pkg_dummy_gen.rand_date(1800), 'sysuser01', pkg_dummy_gen.rand_date(1800), 'sysuser01'
    );
    FOR c IN 1..v_comment_counts(i) LOOP
      INSERT INTO HISTORY_INTERNAL_DETAIL (
        REQUEST_ID, ATTFILE_NUM, FILE_COMMENT, IN_DTM, IN_USERID, CR_DTM, CR_USERID
      ) VALUES (
        v_req_id, c, pkg_dummy_gen.gen_paragraph(150,1400),
        pkg_dummy_gen.rand_date(1700), 'emp'||LPAD(pkg_dummy_gen.rand_int(1,50),4,'0'),
        pkg_dummy_gen.rand_date(1700), 'emp'||LPAD(pkg_dummy_gen.rand_int(1,50),4,'0')
      );
      -- 댓글당 첨부는 TM_ATTFILE PK(DOCUCD,ATTFILE_ID,ATTFILE_NUM) 구조상 최대 1개
      v_has_file := pkg_dummy_gen.rand_int(1,100);
      IF v_has_file <= 40 THEN
        v_name := pkg_dummy_gen.gen_filename(v_ext);
        INSERT INTO TM_ATTFILE (
          DOCUCD, ATTFILE_ID, ATTFILE_NUM, ATTFILE_NM, FILE_PATH, FILE_EXT,
          FILE_SIZE, DOWNLOAD_CNT, CR_USERID, CR_DTM, IN_USERID, IN_DTM
        ) VALUES (
          'C0002014', v_req_id, c, v_name, '/data/dummy/attach/'||v_req_id||'_'||c||'.'||v_ext, v_ext,
          pkg_dummy_gen.rand_int(10000,5000000), pkg_dummy_gen.rand_int(0,500),
          'sysuser01', pkg_dummy_gen.rand_date(1800), 'sysuser01', pkg_dummy_gen.rand_date(900)
        );
      END IF;
    END LOOP;
  END LOOP;
  COMMIT;
END;
/

PROMPT ==== 7. 메뉴4: 심의위원회(HISTORY_DISCUSSION_COMMITTEE) 57건 + 첨부(C0002013) ====
DECLARE
  v_dis_id VARCHAR2(15);
  v_cnt NUMBER;
  v_ext VARCHAR2(20);
  v_name VARCHAR2(150);
  v_result VARCHAR2(100);
  v_noise NUMBER;
BEGIN
  FOR i IN 1..57 LOOP
    v_dis_id := 'DIS'||LPAD(i,7,'0');
    v_result := pkg_dummy_gen.rand_pick(SYS.ODCIVARCHAR2LIST('원안수용','제공불가','항목조정후 제공','보류'));
    -- 정규화 대상 노이즈: 공백이 들쭉날쭉한 실데이터 특성 반영
    v_noise := pkg_dummy_gen.rand_int(1,100);
    IF v_noise <= 15 THEN
      v_result := ' '||v_result;
    ELSIF v_noise <= 30 THEN
      v_result := v_result||' ';
    ELSIF v_noise <= 40 THEN
      v_result := REPLACE(v_result,' ','  ');
    END IF;

    INSERT INTO HISTORY_DISCUSSION_COMMITTEE (
      DISCUSSION_ID, CRP_NO, CRP_NAME, CRP_DEPT_NAME, REQ_PERSON_NAME, REQ_DTM, REQ_TEL,
      TITLE, REQ_MENT, LAW_COMMENT, NOTE, AGENDA_NUMBER, YEAR, SEQ,
      DISCUSSION_RESULT, RESULT_CONTENT, REQ_ITEM, PROC_ITEM, PROC_DTM,
      IN_DTM, IN_USERID, CR_DTM, CR_USERID
    ) VALUES (
      v_dis_id, pkg_dummy_gen.gen_crp_no(), pkg_dummy_gen.gen_org_name(), pkg_dummy_gen.gen_dept_name(),
      pkg_dummy_gen.gen_person_name(), pkg_dummy_gen.rand_yyyymmdd(1800), pkg_dummy_gen.gen_tel(),
      '심의안건 '||pkg_dummy_gen.rand_pick(SYS.ODCIVARCHAR2LIST('통계자료 제공 승인 건','신규 자료 제공 검토','제공 범위 조정 안건','정기 재심의 안건')),
      pkg_dummy_gen.gen_paragraph(300,1900),
      CASE WHEN pkg_dummy_gen.rand_int(1,100) <= 80 THEN pkg_dummy_gen.gen_paragraph(100,720) ELSE NULL END,
      CASE WHEN pkg_dummy_gen.rand_int(1,100) <= 40 THEN pkg_dummy_gen.gen_paragraph(40,450) ELSE NULL END,
      TO_CHAR(2021 + MOD(i,6))||'-'||LPAD(MOD(i,20)+1,3,'0'),
      TO_CHAR(2021 + MOD(i,6)),
      MOD(i,4)+1,
      v_result,
      pkg_dummy_gen.gen_paragraph(150,1400),
      pkg_dummy_gen.gen_paragraph(60,450),
      pkg_dummy_gen.gen_paragraph(60,450),
      CASE WHEN pkg_dummy_gen.rand_int(1,100) <= 85 THEN pkg_dummy_gen.rand_yyyymmdd(1700) ELSE NULL END,
      pkg_dummy_gen.rand_date(1800), 'sysuser01', pkg_dummy_gen.rand_date(1800), 'sysuser01'
    );
    v_cnt := pkg_dummy_gen.rand_int(0,7);
    FOR n IN 1..v_cnt LOOP
      v_name := pkg_dummy_gen.gen_filename(v_ext);
      INSERT INTO TM_ATTFILE (
        DOCUCD, ATTFILE_ID, ATTFILE_NUM, ATTFILE_NM, FILE_PATH, FILE_EXT,
        FILE_SIZE, DOWNLOAD_CNT, CR_USERID, CR_DTM, IN_USERID, IN_DTM
      ) VALUES (
        'C0002013', v_dis_id, n, v_name, '/data/dummy/attach/'||v_dis_id||'_'||n||'.'||v_ext, v_ext,
        pkg_dummy_gen.rand_int(10000,5000000), pkg_dummy_gen.rand_int(0,500),
        'sysuser01', pkg_dummy_gen.rand_date(1800), 'sysuser01', pkg_dummy_gen.rand_date(900)
      );
    END LOOP;
  END LOOP;
  COMMIT;
END;
/

PROMPT ==== 8. 결과 요약 ====
SELECT '기관정보(전체)' AS 구분, COUNT(*) AS 건수 FROM HISTORY_ORGANIZATION_INFO
UNION ALL SELECT '  - 케이스① 메인만(001~025)', COUNT(*) FROM HISTORY_ORGANIZATION_INFO WHERE ORGANIZATION_ID BETWEEN 'ORG0000001' AND 'ORG0000025'
UNION ALL SELECT '  - 케이스② 메인+메인첨부(026~050)', COUNT(*) FROM HISTORY_ORGANIZATION_INFO WHERE ORGANIZATION_ID BETWEEN 'ORG0000026' AND 'ORG0000050'
UNION ALL SELECT '  - 케이스③ 메인+이력+이력첨부(051~075)', COUNT(*) FROM HISTORY_ORGANIZATION_INFO WHERE ORGANIZATION_ID BETWEEN 'ORG0000051' AND 'ORG0000075'
UNION ALL SELECT '  - 케이스④ 전부(076~100)', COUNT(*) FROM HISTORY_ORGANIZATION_INFO WHERE ORGANIZATION_ID BETWEEN 'ORG0000076' AND 'ORG0000100'
UNION ALL SELECT '이력(HISTORY_ORGANIZATION_DETAIL)', COUNT(*) FROM HISTORY_ORGANIZATION_DETAIL
UNION ALL SELECT '외부요청(HISTORY_EXTERNAL_REQUEST)', COUNT(*) FROM HISTORY_EXTERNAL_REQUEST
UNION ALL SELECT '내부자료-글(HISTORY_INTERNAL_INFO)', COUNT(*) FROM HISTORY_INTERNAL_INFO
UNION ALL SELECT '내부자료-댓글(HISTORY_INTERNAL_DETAIL)', COUNT(*) FROM HISTORY_INTERNAL_DETAIL
UNION ALL SELECT '심의위원회(HISTORY_DISCUSSION_COMMITTEE)', COUNT(*) FROM HISTORY_DISCUSSION_COMMITTEE
UNION ALL SELECT '첨부파일 전체(TM_ATTFILE)', COUNT(*) FROM TM_ATTFILE
UNION ALL SELECT '  - C0002010(메뉴1 메인첨부)', COUNT(*) FROM TM_ATTFILE WHERE DOCUCD='C0002010'
UNION ALL SELECT '  - C0002011(메뉴1 이력첨부)', COUNT(*) FROM TM_ATTFILE WHERE DOCUCD='C0002011'
UNION ALL SELECT '  - C0002012(메뉴2 첨부)', COUNT(*) FROM TM_ATTFILE WHERE DOCUCD='C0002012'
UNION ALL SELECT '  - C0002013(메뉴4 첨부)', COUNT(*) FROM TM_ATTFILE WHERE DOCUCD='C0002013'
UNION ALL SELECT '  - C0002014(메뉴3 댓글첨부)', COUNT(*) FROM TM_ATTFILE WHERE DOCUCD='C0002014';
