//! Protocol-only adapters. No app launch, token issuance popup, audio, camera,
//! desktop input, process spawning or implicit broadcast action exists here.
use crate::{
    config::*,
    infra::select,
    transport::{Connection, receive, send},
};
use anyhow::{Result, anyhow, bail};
use base64::{Engine, engine::general_purpose::STANDARD};
use missy_policy::PolicyWebSocket;
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

#[derive(Deserialize)]
#[serde(tag = "action", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum ObsRead {
    Status,
    ListScenes,
}
#[derive(Deserialize)]
#[serde(tag = "action", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum ObsMutate {
    SetSourceText {
        source: String,
        text: String,
    },
    SwitchScene {
        scene: String,
    },
    SetVisibility {
        scene: String,
        item_id: u64,
        visible: bool,
    },
    StartRecording,
    StopRecording,
    StartStreaming,
    StopStreaming,
}
pub(crate) struct Obs {
    pub conn: Connection,
    pub cfg: ObsConfig,
}
impl Obs {
    pub fn validate(cfg: &ObsConfig) -> Result<()> {
        for s in &cfg.scenes {
            label(s)?;
        }
        for s in &cfg.text_sources {
            label(s)?;
        }
        for s in &cfg.scene_items {
            let (scene, id) = s
                .rsplit_once(':')
                .ok_or_else(|| anyhow!("invalid OBS scene:item allowlist"))?;
            if !cfg.scenes.contains(scene) || id.parse::<u64>().is_err() {
                bail!("invalid OBS scene:item allowlist");
            }
        }
        Ok(())
    }
    async fn connect(&self) -> Result<PolicyWebSocket> {
        let mut ws = self.conn.socket().await?;
        let hello = receive(&mut ws, self.conn.cfg.max_output_bytes).await?;
        if hello["op"] != 0 || hello["d"]["rpcVersion"].as_u64().is_none_or(|v| v < 1) {
            bail!("invalid OBS Hello");
        }
        let mut identify = json!({"rpcVersion":1,"eventSubscriptions":0});
        if let Some(auth) = hello["d"].get("authentication") {
            let password = self
                .conn
                .credential()
                .ok_or_else(|| anyhow!("OBS authentication requires configured credential"))?;
            let salt = auth["salt"]
                .as_str()
                .ok_or_else(|| anyhow!("OBS authentication salt missing"))?;
            let challenge = auth["challenge"]
                .as_str()
                .ok_or_else(|| anyhow!("OBS authentication challenge missing"))?;
            if salt.len() > 1024 || challenge.len() > 1024 {
                bail!("OBS authentication challenge exceeds bounds");
            }
            identify["authentication"] = Value::String(obs_auth(password, salt, challenge));
        } else if self.conn.credential().is_some() {
            bail!("OBS expected authenticated server");
        }
        send(&mut ws, json!({"op":1,"d":identify})).await?;
        let identified = receive(&mut ws, self.conn.cfg.max_output_bytes).await?;
        if identified["op"] != 2 || identified["d"]["negotiatedRpcVersion"] != 1 {
            bail!("OBS authentication failed");
        }
        Ok(ws)
    }
    async fn request(&self, ws: &mut PolicyWebSocket, kind: &str, data: Value) -> Result<Value> {
        let id = uuid::Uuid::new_v4().to_string();
        send(
            ws,
            json!({"op":6,"d":{"requestType":kind,"requestId":id,"requestData":data}}),
        )
        .await?;
        let r = receive(ws, self.conn.cfg.max_output_bytes).await?;
        if r["op"] != 7 || r["d"]["requestId"] != id || r["d"]["requestType"] != kind {
            bail!("OBS response correlation failed");
        }
        if r["d"]["requestStatus"]["result"] != true || r["d"]["requestStatus"]["code"] != 100 {
            bail!("OBS request rejected");
        }
        let data = r["d"]
            .get("responseData")
            .cloned()
            .unwrap_or_else(|| json!({}));
        if !data.is_object() {
            bail!("invalid OBS response data");
        }
        Ok(data)
    }
    pub async fn read(&self, a: ObsRead) -> Result<Value> {
        let mut ws = self.connect().await?;
        match a {
            ObsRead::Status => {
                let stream = self.request(&mut ws, "GetStreamStatus", json!({})).await?;
                let record = self.request(&mut ws, "GetRecordStatus", json!({})).await?;
                Ok(
                    json!({"stream":select(&stream,&["outputActive","outputReconnecting","outputDuration","outputBytes"]),
                    "record":select(&record,&["outputActive","outputPaused","outputDuration","outputBytes"])}),
                )
            }
            ObsRead::ListScenes => {
                let r = self.request(&mut ws, "GetSceneList", json!({})).await?;
                let scenes = r["scenes"]
                    .as_array()
                    .ok_or_else(|| anyhow!("invalid OBS scenes response"))?;
                let scenes: Vec<Value> = scenes
                    .iter()
                    .filter(|s| {
                        s["sceneName"]
                            .as_str()
                            .is_some_and(|n| self.cfg.scenes.contains(n))
                    })
                    .map(|v| select(v, &["sceneName", "sceneIndex"]))
                    .collect();
                let current = r["currentProgramSceneName"]
                    .as_str()
                    .filter(|s| self.cfg.scenes.contains(*s));
                Ok(json!({"scenes":scenes,"current_scene":current}))
            }
        }
    }
    pub async fn mutate(&self, a: ObsMutate) -> Result<Value> {
        let (action, kind, data) = match a {
            ObsMutate::SetSourceText { source, text } => {
                if !self.cfg.text_sources.contains(&source)
                    || text.len() > 8192
                    || text.contains('\0')
                {
                    bail!("OBS text source/content denied");
                }
                (
                    ObsAction::SetSourceText,
                    "SetInputSettings",
                    json!({"inputName":source,"inputSettings":{"text":text},"overlay":true}),
                )
            }
            ObsMutate::SwitchScene { scene } => {
                if !self.cfg.scenes.contains(&scene) {
                    bail!("OBS scene denied");
                }
                (
                    ObsAction::SwitchScene,
                    "SetCurrentProgramScene",
                    json!({"sceneName":scene}),
                )
            }
            ObsMutate::SetVisibility {
                scene,
                item_id,
                visible,
            } => {
                if !self.cfg.scenes.contains(&scene)
                    || !self.cfg.scene_items.contains(&format!("{scene}:{item_id}"))
                {
                    bail!("OBS scene item denied");
                }
                (
                    ObsAction::SetVisibility,
                    "SetSceneItemEnabled",
                    json!({"sceneName":scene,"sceneItemId":item_id,"sceneItemEnabled":visible}),
                )
            }
            ObsMutate::StartRecording => (ObsAction::StartRecording, "StartRecord", json!({})),
            ObsMutate::StopRecording => (ObsAction::StopRecording, "StopRecord", json!({})),
            ObsMutate::StartStreaming => (ObsAction::StartStreaming, "StartStream", json!({})),
            ObsMutate::StopStreaming => (ObsAction::StopStreaming, "StopStream", json!({})),
        };
        if !self.cfg.actions.contains(&action) {
            bail!("OBS mutation action denied");
        }
        let mut ws = self.connect().await?;
        self.request(&mut ws, kind, data).await?;
        Ok(json!({"acknowledged":true,"action":action}))
    }
}
pub(crate) fn obs_auth(password: &str, salt: &str, challenge: &str) -> String {
    let secret = STANDARD.encode(Sha256::digest(format!("{password}{salt}")));
    STANDARD.encode(Sha256::digest(format!("{secret}{challenge}")))
}

#[derive(Deserialize)]
#[serde(tag = "action", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum VtubeRead {
    Status,
    ListModels,
    ListHotkeys,
}
#[derive(Deserialize)]
#[serde(tag = "action", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum VtubeMutate {
    LoadModel { model: String },
    TriggerHotkey { hotkey: String },
    SetParameter { parameter: String, value: f64 },
}
pub(crate) struct Vtube {
    pub conn: Connection,
    pub cfg: VtubeConfig,
}
impl Vtube {
    pub fn validate(cfg: &VtubeConfig) -> Result<()> {
        label(&cfg.plugin_name)?;
        label(&cfg.plugin_developer)?;
        if cfg.connection.token_env.is_none() {
            bail!(
                "VTube Studio requires a pre-provisioned token environment name; token popups disabled"
            );
        }
        for s in cfg
            .models
            .iter()
            .chain(cfg.hotkeys.iter())
            .chain(cfg.parameters.keys())
        {
            identifier(s)?;
        }
        for r in cfg.parameters.values() {
            if !r.min.is_finite()
                || !r.max.is_finite()
                || r.min > r.max
                || r.min < -1_000_000.0
                || r.max > 1_000_000.0
            {
                bail!("VTube Studio parameter range invalid");
            }
        }
        Ok(())
    }
    pub(crate) async fn request(
        &self,
        ws: &mut PolicyWebSocket,
        kind: &str,
        data: Value,
    ) -> Result<Value> {
        let id = uuid::Uuid::new_v4().to_string();
        send(ws,json!({"apiName":"VTubeStudioPublicAPI","apiVersion":"1.0","requestID":id,"messageType":kind,"data":data})).await?;
        let r = receive(ws, self.conn.cfg.max_output_bytes).await?;
        let expected = kind
            .strip_suffix("Request")
            .ok_or_else(|| anyhow!("invalid VTube request type"))?
            .to_owned()
            + "Response";
        if r["apiName"] != "VTubeStudioPublicAPI"
            || r["apiVersion"] != "1.0"
            || r["requestID"] != id
        {
            bail!("VTube Studio response correlation failed");
        }
        if r["messageType"] != expected {
            bail!("VTube Studio request rejected or response type mismatch");
        }
        if !r["data"].is_object() {
            bail!("invalid VTube Studio data");
        }
        Ok(r["data"].clone())
    }
    pub(crate) async fn connect(&self) -> Result<PolicyWebSocket> {
        let mut ws = self.conn.socket().await?;
        let token = self
            .conn
            .credential()
            .ok_or_else(|| anyhow!("VTube Studio credential unavailable"))?;
        let r = self
            .request(
                &mut ws,
                "AuthenticationRequest",
                json!({"pluginName":self.cfg.plugin_name,
            "pluginDeveloper":self.cfg.plugin_developer,"authenticationToken":token}),
            )
            .await?;
        if r["authenticated"] != true {
            bail!("VTube Studio authentication rejected");
        }
        Ok(ws)
    }
    pub async fn read(&self, a: VtubeRead) -> Result<Value> {
        let mut ws = self.connect().await?;
        match a {
            VtubeRead::Status => {
                let r = self
                    .request(&mut ws, "CurrentModelRequest", json!({}))
                    .await?;
                if r["modelID"]
                    .as_str()
                    .is_some_and(|id| self.cfg.models.contains(id))
                {
                    Ok(select(&r, &["modelLoaded", "modelID", "modelName"]))
                } else {
                    Ok(json!({"modelLoaded":r["modelLoaded"],"model_visible":false}))
                }
            }
            VtubeRead::ListModels => {
                let r = self
                    .request(&mut ws, "AvailableModelsRequest", json!({}))
                    .await?;
                let models = r["availableModels"]
                    .as_array()
                    .ok_or_else(|| anyhow!("invalid VTube Studio models list"))?;
                Ok(
                    json!({"models":models.iter().filter(|v| v["modelID"].as_str().is_some_and(|id| self.cfg.models.contains(id)))
                    .map(|v|select(v,&["modelID","modelName"])).collect::<Vec<_>>()}),
                )
            }
            VtubeRead::ListHotkeys => {
                let r = self
                    .request(&mut ws, "HotkeysInCurrentModelRequest", json!({}))
                    .await?;
                let hotkeys = r["availableHotkeys"]
                    .as_array()
                    .ok_or_else(|| anyhow!("invalid VTube Studio hotkeys list"))?;
                Ok(
                    json!({"hotkeys":hotkeys.iter().filter(|v| v["hotkeyID"].as_str().is_some_and(|id| self.cfg.hotkeys.contains(id)))
                    .map(|v|select(v,&["hotkeyID","name","type"])).collect::<Vec<_>>()}),
                )
            }
        }
    }
    pub async fn mutate(&self, a: VtubeMutate) -> Result<Value> {
        let (kind, data) = match a {
            VtubeMutate::LoadModel { model } => {
                if !self.cfg.models.contains(&model) {
                    bail!("VTube Studio model denied");
                }
                ("ModelLoadRequest", json!({"modelID":model}))
            }
            VtubeMutate::TriggerHotkey { hotkey } => {
                if !self.cfg.hotkeys.contains(&hotkey) {
                    bail!("VTube Studio hotkey denied");
                }
                ("HotkeyTriggerRequest", json!({"hotkeyID":hotkey}))
            }
            VtubeMutate::SetParameter { parameter, value } => {
                let r = self
                    .cfg
                    .parameters
                    .get(&parameter)
                    .ok_or_else(|| anyhow!("VTube Studio parameter denied"))?;
                if !value.is_finite() || value < r.min || value > r.max {
                    bail!("VTube Studio parameter value outside approved range");
                }
                (
                    "InjectParameterDataRequest",
                    json!({"faceFound":false,"mode":"set","parameterValues":[{"id":parameter,"value":value,"weight":1.0}]}),
                )
            }
        };
        let mut ws = self.connect().await?;
        self.request(&mut ws, kind, data).await?;
        Ok(json!({"acknowledged":true}))
    }
}
