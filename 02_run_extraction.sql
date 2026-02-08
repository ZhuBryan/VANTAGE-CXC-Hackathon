USE ROLE SYSADMIN;
USE DATABASE ECO_AGENT_DB;
USE SCHEMA AUDIT_LOGS;

SET report_name = 'tesla-2023.pdf';

INSERT INTO SATELLITE_AUDIT (COMPANY_NAME, ASSET_NAME, CLAIM_FOR_SATELLITE, EXPECTED_PARAMETER, COORDINATE_HINT)
SELECT 
    'Tesla',
    'KCC Operation (DRC)',
    SNOWFLAKE.CORTEX.COMPLETE('mistral-large2', 'Extract the claim about satellite monitoring resolution and update frequency at the KCC mine. Return ONLY the claim: ' || TO_VARCHAR(SNOWFLAKE.CORTEX.PARSE_DOCUMENT(@ESG_REPORTS_STAGE, $report_name, {'mode': 'LAYOUT'}):content)),
    'Mine',
    'Lualaba, DRC';

INSERT INTO SATELLITE_AUDIT (COMPANY_NAME, ASSET_NAME, CLAIM_FOR_SATELLITE, EXPECTED_PARAMETER, COORDINATE_HINT)
SELECT 
    'Tesla',
    'Corpus Christi Refinery',
    SNOWFLAKE.CORTEX.COMPLETE('mistral-large2', 'Find the claim about the byproduct generated at the Corpus Christi refinery and what it will be used for. Return ONLY the claim: ' || TO_VARCHAR(SNOWFLAKE.CORTEX.PARSE_DOCUMENT(@ESG_REPORTS_STAGE, 'tesla-2023.pdf', {'mode': 'LAYOUT'}):content)),
    'Industrial',
    'Robstown, TX';

INSERT INTO SATELLITE_AUDIT (COMPANY_NAME, ASSET_NAME, CLAIM_FOR_SATELLITE, EXPECTED_PARAMETER, COORDINATE_HINT)
SELECT 
    'Tesla',
    'Giga Shanghai',
    SNOWFLAKE.CORTEX.COMPLETE('mistral-large2', 'Extract the percentage of energy reduction at Giga Shanghai compared to Fremont from this text. Return ONLY the claim sentence: ' || TO_VARCHAR(SNOWFLAKE.CORTEX.PARSE_DOCUMENT(@ESG_REPORTS_STAGE, 'tesla-2023.pdf', {'mode': 'LAYOUT'}):content)),
    'Industrial',
    'Pudong, Shanghai';

-- SELECT * FROM SATELLITE_AUDIT;