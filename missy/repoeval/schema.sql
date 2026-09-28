-- Explicitly applied PostgreSQL schema; application startup never runs migrations.
CREATE TABLE IF NOT EXISTS projects (
  project_id text PRIMARY KEY, created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS runs (
  project_id text NOT NULL REFERENCES projects(project_id),
  run_id text NOT NULL CHECK (run_id ~ '^run-[A-Za-z0-9_-]{8,128}$'),
  manifest jsonb NOT NULL,
  manifest_sha256 char(64) NOT NULL CHECK (manifest_sha256 ~ '^[0-9a-f]{64}$'),
  idempotency_key_sha256 char(64) NOT NULL CHECK (idempotency_key_sha256 ~ '^[0-9a-f]{64}$'),
  state text NOT NULL CHECK (state IN ('planned','submitted','running','collecting','verified','failed','cancelled','incomparable')),
  state_detail jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(), updated_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY(project_id, run_id), UNIQUE(project_id, idempotency_key_sha256)
);
CREATE TABLE IF NOT EXISTS run_state_events (
  event_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  project_id text NOT NULL, run_id text NOT NULL, from_state text, to_state text NOT NULL,
  details jsonb NOT NULL DEFAULT '{}'::jsonb, created_at timestamptz NOT NULL DEFAULT now(),
  FOREIGN KEY(project_id, run_id) REFERENCES runs(project_id, run_id)
);
CREATE TABLE IF NOT EXISTS artifact_collection_allowlist (
  project_id text NOT NULL, run_id text NOT NULL, collection_key text NOT NULL,
  kind text NOT NULL, required boolean NOT NULL DEFAULT false, created_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY(project_id, run_id, collection_key),
  FOREIGN KEY(project_id, run_id) REFERENCES runs(project_id, run_id),
  CHECK (collection_key ~ '^[a-z0-9][a-z0-9._-]{0,127}$'), CHECK (kind ~ '^[a-z0-9][a-z0-9._-]{0,63}$')
);
CREATE TABLE IF NOT EXISTS artifacts (
  project_id text NOT NULL, artifact_id text NOT NULL, run_id text NOT NULL, collection_key text NOT NULL,
  kind text NOT NULL, uri text NOT NULL, sha256 char(64) NOT NULL CHECK (sha256 ~ '^[0-9a-f]{64}$'),
  size_bytes bigint NOT NULL CHECK (size_bytes >= 0), media_type text NOT NULL, producer jsonb NOT NULL,
  classification text NOT NULL CHECK (classification IN ('public','internal','repository-sensitive','restricted')),
  retention_class text NOT NULL, scan_state text NOT NULL CHECK (scan_state IN ('pending','clean','quarantined','failed')),
  redaction_state text NOT NULL CHECK (redaction_state IN ('not-required','pending','complete','failed')),
  created_at timestamptz NOT NULL, expires_at timestamptz, deleted_at timestamptz, manifest jsonb NOT NULL,
  PRIMARY KEY(project_id, artifact_id),
  FOREIGN KEY(project_id, run_id, collection_key) REFERENCES artifact_collection_allowlist(project_id, run_id, collection_key)
);
CREATE INDEX IF NOT EXISTS artifacts_retention_due ON artifacts(project_id, expires_at) WHERE deleted_at IS NULL AND expires_at IS NOT NULL;
CREATE INDEX IF NOT EXISTS artifacts_run_catalog ON artifacts(project_id, run_id, created_at, artifact_id) WHERE deleted_at IS NULL;
CREATE TABLE IF NOT EXISTS artifact_tombstones (
  project_id text NOT NULL, artifact_id text NOT NULL, run_id text NOT NULL, sha256 char(64) NOT NULL,
  disposition text NOT NULL CHECK (disposition IN ('expired','deleted','quarantined')),
  reason text NOT NULL, deleted_at timestamptz NOT NULL,
  PRIMARY KEY(project_id, artifact_id), FOREIGN KEY(project_id, artifact_id) REFERENCES artifacts(project_id, artifact_id)
);
CREATE TABLE IF NOT EXISTS artifact_retention_policies (
  project_id text NOT NULL REFERENCES projects(project_id), retention_class text NOT NULL,
  retention_seconds bigint NOT NULL CHECK (retention_seconds >= 0), classification text NOT NULL,
  updated_at timestamptz NOT NULL DEFAULT now(), PRIMARY KEY(project_id, retention_class)
);
CREATE OR REPLACE FUNCTION foundry_reject_run_identity_update() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  IF NEW.project_id <> OLD.project_id OR NEW.run_id <> OLD.run_id OR NEW.manifest <> OLD.manifest OR
     NEW.manifest_sha256 <> OLD.manifest_sha256 OR NEW.idempotency_key_sha256 <> OLD.idempotency_key_sha256 THEN
    RAISE EXCEPTION 'run identity and manifest are immutable';
  END IF;
  RETURN NEW;
END $$;
DROP TRIGGER IF EXISTS runs_immutable_identity ON runs;
CREATE TRIGGER runs_immutable_identity BEFORE UPDATE ON runs FOR EACH ROW EXECUTE FUNCTION foundry_reject_run_identity_update();
