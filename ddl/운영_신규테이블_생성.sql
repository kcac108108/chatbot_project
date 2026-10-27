-- 운영 서버 배포 전 CHATBOT 스키마에 신규로 생성해야 하는 테이블 2개.
-- 이력관리시스템 원본 테이블(HISTORY_*, TM_ATTFILE)은 운영에 이미 존재하여
-- DB링크(@STATWEB)로 조회만 하므로 이 파일에는 포함하지 않음.
-- (전체 DDL은 "이력관리시스템 관련 테이블 생성 스크립트.sql" 8~9번 항목과 동일 내용)

-- 1. 정형 데이터(메뉴1~4 본문) 증분 적재 실행 기록
CREATE TABLE CHATBOT_INGEST_LOG
(
  RUN_AT      DATE            NOT NULL,
  CHUNK_COUNT NUMBER          NOT NULL
);

COMMENT ON TABLE CHATBOT_INGEST_LOG IS '챗봇 적재 배치 실행 기록 — 증분 적재 시 마지막 실행 시각 조회용';
COMMENT ON COLUMN CHATBOT_INGEST_LOG.RUN_AT IS '적재 배치 실행(완료) 시각';
COMMENT ON COLUMN CHATBOT_INGEST_LOG.CHUNK_COUNT IS '해당 실행에서 적재된 청크 수';

-- 2. 첨부파일 증분 적재 실행 기록 (메뉴별로 따로 추적)
CREATE TABLE CHATBOT_ATTACHMENT_INGEST_LOG
(
  DOCUCD      VARCHAR2(10 BYTE) NOT NULL,
  RUN_AT      DATE              NOT NULL,
  CHUNK_COUNT NUMBER            NOT NULL
);

COMMENT ON TABLE CHATBOT_ATTACHMENT_INGEST_LOG IS '챗봇 첨부파일 증분 적재 실행 기록 — DOCUCD(메뉴)별 마지막 실행 시각 조회용';
COMMENT ON COLUMN CHATBOT_ATTACHMENT_INGEST_LOG.DOCUCD IS '첨부파일구분(TM_ATTFILE.DOCUCD와 동일)';
COMMENT ON COLUMN CHATBOT_ATTACHMENT_INGEST_LOG.RUN_AT IS '적재 배치 실행(완료) 시각';
COMMENT ON COLUMN CHATBOT_ATTACHMENT_INGEST_LOG.CHUNK_COUNT IS '해당 실행에서 적재된 청크 수';
