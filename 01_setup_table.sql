-- Run once to create the table

USE DATABASE ECO_AGENT_DB;
USE SCHEMA AUDIT_LOGS;

CREATE OR REPLACE TABLE SATELLITE_AUDIT (
    COMPANY_NAME STRING COMMENT 'Name of the audited firm',
    ASSET_NAME STRING COMMENT 'The physical location or facility to be checked',
    CLAIM_FOR_SATELLITE STRING COMMENT 'The specific claim to be falsified via satellite',
    EXPECTED_PARAMETER STRING COMMENT 'API parameter: Forest, Mine, Water, Industrial',
    COORDINATE_HINT STRING COMMENT 'GPS or City hint for Satellite Map',
    AUDIT_DATE DATE DEFAULT CURRENT_DATE()
);