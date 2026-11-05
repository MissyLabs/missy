CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL) STRICT;
CREATE TABLE memories(
    rowid INTEGER PRIMARY KEY,
    tenant TEXT NOT NULL,
    user TEXT NOT NULL,
    id TEXT NOT NULL,
    content TEXT NOT NULL,
    source_scope TEXT NOT NULL,
    checksum TEXT NOT NULL,
    UNIQUE(tenant,user,id)
) STRICT;
CREATE VIRTUAL TABLE personal_fts USING fts5(content, tokenize='unicode61');
