//! document schema: field declarations, misc types for future extension

use std::collections::HashMap;

/// one document as returned by a get-by-id / search result store read
#[derive(Debug, Clone, Default)]
pub struct StoredDoc {
    /// external id, when the doc carries one
    pub id: Option<String>,
    /// (field_id, text) pairs, sorted by field id
    pub fields: Vec<(u32, String)>,
    /// stored-only metadata, when the doc carries it
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FieldKind {
    #[allow(dead_code)]
    Text,
    /// declared now, indexed later; gates schema validation for the embedding extension
    #[allow(dead_code)]
    Vector { dim: u32 },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Field {
    pub name: String,
    #[allow(dead_code)]
    pub kind: FieldKind,
    /// relative weight applied to BM25 contributions from this field
    pub boost: f32,
}

#[derive(Debug, Clone, Default)]
pub struct Schema {
    fields: Vec<Field>,
    by_name: HashMap<String, u32>,
}

impl Schema {
    pub fn empty() -> Self {
        Self::default()
    }

    pub fn with_text(mut self, name: &str, boost: f32) -> Self {
        self.add_text(name, boost);
        self
    }

    pub fn add_text(&mut self, name: &str, boost: f32) {
        let f = Field {
            name: name.to_string(),
            kind: FieldKind::Text,
            boost,
        };
        self.fields.push(f);
        self.by_name
            .insert(name.to_string(), self.fields.len() as u32 - 1);
    }

    pub fn field_id(&self, name: &str) -> Option<u32> {
        self.by_name.get(name).copied()
    }

    pub fn field(&self, id: u32) -> Option<&Field> {
        self.fields.get(id as usize)
    }

    pub fn fields(&self) -> &[Field] {
        &self.fields
    }

    pub fn boost(&self, id: u32) -> f32 {
        self.fields.get(id as usize).map(|f| f.boost).unwrap_or(1.0)
    }

    pub fn num_fields(&self) -> usize {
        self.fields.len()
    }
}

/// one document: optional external id, stored-only metadata, field -> text.
/// Unknown fields are rejected at add time.
#[derive(Debug, Clone, Default)]
pub struct Document {
    /// user-assigned identity (unique across the live index); None = implicit internal id
    pub id: Option<String>,
    /// arbitrary JSON, stored only (never tokenized or scored)
    pub metadata: Option<serde_json::Value>,
    pub fields: HashMap<String, String>,
}

impl serde::Serialize for Document {
    /// wire shape: {"id": ..., "metadata": ..., "<field>": "<text>", ...}
    /// (the ingest parsers read flat `text`; internal fields are flattened)
    fn serialize<S: serde::Serializer>(&self, ser: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        let mut map = ser.serialize_map(None)?;
        if let Some(id) = &self.id {
            map.serialize_entry("id", id)?;
        }
        if let Some(meta) = &self.metadata {
            map.serialize_entry("metadata", meta)?;
        }
        for (k, v) in &self.fields {
            map.serialize_entry(k, v)?;
        }
        map.end()
    }
}

impl<'de> serde::Deserialize<'de> for Document {
    fn deserialize<D: serde::Deserializer<'de>>(de: D) -> Result<Self, D::Error> {
        use serde::de::Error;
        let v = serde_json::Value::deserialize(de)?;
        let obj = v
            .as_object()
            .ok_or_else(|| D::Error::custom("doc must be an object"))?;
        let mut doc = Document::default();
        if let Some(id) = obj.get("id") {
            let id = id
                .as_str()
                .ok_or_else(|| D::Error::custom("id must be a string"))?;
            doc.id = Some(id.to_owned());
        }
        if let Some(meta) = obj.get("metadata") {
            if !meta.is_null() {
                doc.metadata = Some(meta.clone());
            }
        }
        for (k, val) in obj {
            if k == "id" || k == "metadata" {
                continue;
            }
            let text = val
                .as_str()
                .ok_or_else(|| D::Error::custom(format!("`{k}` must be a string")))?;
            doc.fields.insert(k.clone(), text.to_owned());
        }
        Ok(doc)
    }
}

impl Document {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn text(mut self, field: &str, value: impl Into<String>) -> Self {
        self.fields.insert(field.to_string(), value.into());
        self
    }

    pub fn id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    pub fn metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = Some(metadata);
        self
    }
}
