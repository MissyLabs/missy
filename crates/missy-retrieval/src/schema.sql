CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL) STRICT;
CREATE TABLE documents(
 scope TEXT NOT NULL, id TEXT NOT NULL, text TEXT NOT NULL, provenance TEXT NOT NULL,
 hash TEXT NOT NULL, PRIMARY KEY(scope,id)
) STRICT;
CREATE TABLE chunks(
 scope TEXT NOT NULL, doc_id TEXT NOT NULL, ordinal INTEGER NOT NULL CHECK(ordinal>=0),
 start_byte INTEGER NOT NULL CHECK(start_byte>=0), end_byte INTEGER NOT NULL CHECK(end_byte>start_byte),
 vector BLOB NOT NULL, checksum TEXT NOT NULL,
 PRIMARY KEY(scope,doc_id,ordinal),
 FOREIGN KEY(scope,doc_id) REFERENCES documents(scope,id) ON DELETE CASCADE
) STRICT;
CREATE TABLE nodes(
 scope TEXT NOT NULL, id TEXT NOT NULL, name TEXT NOT NULL,
 entity_type TEXT NOT NULL, provenance TEXT NOT NULL, PRIMARY KEY(scope,id)
) STRICT;
CREATE TABLE edges(
 scope TEXT NOT NULL, source_id TEXT NOT NULL, target_id TEXT NOT NULL,
 relationship TEXT NOT NULL, provenance TEXT NOT NULL,
 PRIMARY KEY(scope,source_id,target_id,relationship),
 FOREIGN KEY(scope,source_id) REFERENCES nodes(scope,id) ON DELETE CASCADE,
 FOREIGN KEY(scope,target_id) REFERENCES nodes(scope,id) ON DELETE CASCADE
) STRICT;
CREATE INDEX edges_target ON edges(scope,target_id);
