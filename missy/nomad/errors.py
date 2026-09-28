"""Nomad integration exceptions."""


class NomadError(RuntimeError):
    """Base error for a safe, user-displayable Nomad failure."""


class NomadConfigurationError(NomadError):
    """The Nomad integration or its credential bundle is not usable."""


class NomadAuthorizationError(NomadError):
    """The configured ACL identity denied an operation."""


class NomadOwnershipError(NomadError):
    """A mutation targeted a job Missy cannot prove she owns."""


class NomadValidationError(NomadError):
    """A workload request or job specification is invalid."""


class NomadCommandError(NomadError):
    """The Nomad CLI returned a definite failure."""


class NomadMutationUnknown(NomadError):
    """A mutating CLI call timed out, leaving its server-side effect unknown."""
