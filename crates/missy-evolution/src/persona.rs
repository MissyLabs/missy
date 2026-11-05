//! Experiments are immutable candidate documents. Assignments and untrusted
//! evaluation scores cannot promote themselves or mutate the active prompt.
use crate::{EvolutionRegistry, binding, operator, scope_key};
use anyhow::{Context, Result, ensure};
use missy_core::Scope;
use missy_governance::{ApprovalRegistry, DocumentKind, DocumentProposal};
use rusqlite::{OptionalExtension, params};
use serde::{Deserialize, Serialize};

#[derive(Clone)]
pub struct ExperimentRegistry {
    store: EvolutionRegistry,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct PersonaVariant {
    pub name: String,
    pub prompt: String,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct PersonaExperiment {
    pub id: String,
    pub scope_hash: String,
    pub variants: Vec<PersonaVariant>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExperimentComparison {
    pub variant: PersonaVariant,
    pub observations: u64,
    pub successes: u64,
    pub refusals: u64,
    pub success_rate: f64,
    pub refusal_rate: f64,
    pub trusted_for_promotion: bool,
}
impl ExperimentRegistry {
    pub fn new(store: EvolutionRegistry) -> Self {
        Self { store }
    }
    pub fn create(
        &self,
        scope: &Scope,
        variants: Vec<PersonaVariant>,
    ) -> Result<(PersonaExperiment, String)> {
        ensure!(
            (2..=16).contains(&variants.len()),
            "2..16 variants required"
        );
        let mut names = std::collections::BTreeSet::new();
        for v in &variants {
            ensure!(
                !v.name.is_empty()
                    && v.name.len() <= 128
                    && names.insert(v.name.clone())
                    && !v.prompt.trim().is_empty()
                    && v.prompt.len() <= 1024 * 1024
                    && !v.prompt.contains('\0'),
                "invalid/duplicate persona variant"
            );
        }
        let e = PersonaExperiment {
            id: uuid::Uuid::new_v4().to_string(),
            scope_hash: scope_key(scope)?,
            variants,
        };
        let hash = binding(&e)?;
        self.store.connect()?.execute(
            "INSERT INTO experiment(scope,id,payload,hash) VALUES(?1,?2,?3,?4)",
            params![e.scope_hash, e.id, serde_json::to_string(&e)?, hash],
        )?;
        Ok((e, hash))
    }
    pub fn experiment(&self, scope: &Scope, id: &str) -> Result<(PersonaExperiment, String)> {
        let key = scope_key(scope)?;
        let (payload, hash): (String, String) = self
            .store
            .connect()?
            .query_row(
                "SELECT payload,hash FROM experiment WHERE scope=?1 AND id=?2",
                params![key, id],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()?
            .context("experiment not found in scope")?;
        let e: PersonaExperiment = serde_json::from_str(&payload)?;
        ensure!(
            e.scope_hash == key && e.id == id && binding(&e)? == hash,
            "experiment integrity mismatch"
        );
        Ok((e, hash))
    }
    /// Deterministic and persisted. Principal is always derived from trusted
    /// scope tenant/user/channel/session, never model-controlled conversation text.
    pub fn assign(&self, scope: &Scope, id: &str) -> Result<PersonaVariant> {
        let (e, _) = self.experiment(scope, id)?;
        let key = scope_key(scope)?;
        let principal = binding(&(
            "principal-v1",
            &scope.tenant,
            &scope.user,
            &scope.channel,
            &scope.session,
        ))?;
        let seed = binding(&("persona-ab-v1", &e.id, &principal))?;
        let n = u64::from_str_radix(&seed[..16], 16)? as usize % e.variants.len();
        let c = self.store.connect()?;
        c.execute("INSERT OR IGNORE INTO experiment_assignment(scope,experiment,principal,variant) VALUES(?1,?2,?3,?4)",params![key,id,principal,n as i64])?;
        let selected:i64=c.query_row("SELECT variant FROM experiment_assignment WHERE scope=?1 AND experiment=?2 AND principal=?3",params![key,id,principal],|r|r.get(0))?;
        Ok(e.variants
            .get(selected as usize)
            .context("invalid assignment")?
            .clone())
    }
    /// Event IDs are deduplicated. Outcomes are observations, not authority.
    pub fn record_outcome(
        &self,
        scope: &Scope,
        id: &str,
        event_id: &str,
        success: bool,
        refused: bool,
    ) -> Result<()> {
        ensure!(
            !event_id.is_empty() && event_id.len() <= 256,
            "bounded event identifier required"
        );
        let selected = self.assign(scope, id)?;
        let (e, _) = self.experiment(scope, id)?;
        let n = e
            .variants
            .iter()
            .position(|v| v == &selected)
            .context("variant")?;
        let c = self.store.connect()?;
        let key = scope_key(scope)?;
        c.execute("INSERT OR IGNORE INTO experiment_outcome(scope,experiment,event,variant,success,refused) VALUES(?1,?2,?3,?4,?5,?6)",params![key,id,event_id,n as i64,success,refused])?;
        Ok(())
    }
    pub fn compare(&self, scope: &Scope, id: &str) -> Result<Vec<ExperimentComparison>> {
        let (e, _) = self.experiment(scope, id)?;
        let c = self.store.connect()?;
        let key = scope_key(scope)?;
        e.variants.into_iter().enumerate().map(|(i,variant)|{
            let (n,yes,no):(u64,u64,u64)=c.query_row("SELECT COUNT(*),COALESCE(SUM(success),0),COALESCE(SUM(refused),0) FROM experiment_outcome WHERE scope=?1 AND experiment=?2 AND variant=?3",params![key,id,i as i64],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?)))?;
            Ok(ExperimentComparison {variant,observations:n,successes:yes,refusals:no,success_rate:if n==0{0.0}else{yes as f64/n as f64},refusal_rate:if n==0{0.0}else{no as f64/n as f64},trusted_for_promotion:false})
        }).collect()
    }
    /// Operator chooses an exact immutable variant. This produces only a pending
    /// governance proposal. `ApprovalRegistry::decide_document` is the separate
    /// final human promotion gate, preserving base-version/context checks and
    /// runtime PromptSnapshot semantics. No parallel active prompt store exists.
    #[allow(clippy::too_many_arguments)]
    pub fn propose_promotion(
        &self,
        scope: &Scope,
        id: &str,
        exact_experiment_hash: &str,
        variant_name: &str,
        governance: &ApprovalRegistry,
        base_version: i64,
        context_hash: &str,
        operator_id: &str,
        confirm: bool,
    ) -> Result<DocumentProposal> {
        operator(scope, operator_id, confirm)?;
        let (e, h) = self.experiment(scope, id)?;
        ensure!(h == exact_experiment_hash, "exact experiment hash required");
        let selected = e
            .variants
            .iter()
            .find(|v| v.name == variant_name)
            .context("unknown variant")?;
        governance.propose_document(
            scope,
            DocumentKind::Persona,
            base_version,
            &selected.prompt,
            context_hash,
        )
    }
}

/// First-run answers are inert data, never instructions to configure providers,
/// write credentials, seed memory, spawn processes or edit live config files.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct HatchingAnswers {
    pub name: String,
    pub tone: String,
    pub purpose: String,
    pub provider_label: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HatchingProposal {
    pub answers: HatchingAnswers,
    pub config_proposal: serde_json::Value,
    pub persona_proposal: DocumentProposal,
    pub answer_hash: String,
}
pub fn propose_hatching(
    scope: &Scope,
    answers: HatchingAnswers,
    governance: &ApprovalRegistry,
    context_hash: &str,
) -> Result<HatchingProposal> {
    for s in [
        &answers.name,
        &answers.tone,
        &answers.purpose,
        &answers.provider_label,
    ] {
        ensure!(
            !s.trim().is_empty() && s.len() <= 4096 && !s.contains('\0'),
            "invalid hatching answer"
        );
    }
    let answer_hash = binding(&(scope, &answers))?;
    let prompt = format!(
        "Name: {}\nTone: {}\nPurpose: {}\n",
        answers.name, answers.tone, answers.purpose
    );
    let base = governance
        .document(scope, DocumentKind::Persona)?
        .map_or(0, |d| d.version);
    let persona_proposal =
        governance.propose_document(scope, DocumentKind::Persona, base, &prompt, context_hash)?;
    let config_proposal = serde_json::json!({"provider_label":answers.provider_label,"requires_operator_configuration":true,"contains_credentials":false});
    Ok(HatchingProposal {
        answers,
        config_proposal,
        persona_proposal,
        answer_hash,
    })
}
