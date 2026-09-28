# Build only after resolving a real audited digest for a trusted Python base
# with the reviewed jsonschema dependency installed. No network pip install.
# Never replace this with a floating tag or a fabricated digest.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ARG BASE_IMAGE
RUN python -c 'import re,sys; assert re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/:+-]*@sha256:[0-9a-f]{64}", sys.argv[1]), "verified base digest required"' "$BASE_IMAGE"

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/opt/repoeval
WORKDIR /opt/repoeval
RUN python -c "import jsonschema; from jsonschema import Draft202012Validator"
COPY --chown=65532:65532 missy/__init__.py /opt/repoeval/missy/__init__.py
COPY --chown=65532:65532 missy/repoeval/__init__.py missy/repoeval/api.py \
    missy/repoeval/control.py missy/repoeval/scanner.py \
    /opt/repoeval/missy/repoeval/
COPY --chown=65532:65532 missy/repoeval/entrypoint.py missy/repoeval/worker.py \
    missy/repoeval/contracts.py missy/repoeval/provider.py \
    missy/repoeval/evaluation.py /opt/repoeval/missy/repoeval/
USER 65532:65532
ENTRYPOINT ["python", "-m", "missy.repoeval.entrypoint"]
