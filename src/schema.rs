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
