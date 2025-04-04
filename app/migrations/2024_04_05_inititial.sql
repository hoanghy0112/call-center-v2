CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE call (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    voice TEXT,
    temperature FLOAT,
    system_prompt TEXT,
    language TEXT,
    use_lipsync BOOLEAN,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
