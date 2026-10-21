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

/// similarity metric for a vector field (defined in the vector core so that
/// module stays free of engine types)
pub use crate::vector::Metric;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FieldKind {
    Text,
    /// a vector (embedding) field: fixed dimension, one similarity metric
    Vector {
        dim: u32,
        metric: Metric,
    },
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

    /// declare a vector (embedding) field: fixed dimension + similarity metric.
    /// Vector fields share the field-id space with text fields (uniform
    /// `field_id` lookup) but carry NO postings and NO text norms.
    pub fn add_vector(&mut self, name: &str, dim: u32, metric: Metric) {
        let f = Field {
            name: name.to_string(),
            kind: FieldKind::Vector { dim, metric },
            boost: 1.0,
        };
        self.fields.push(f);
        self.by_name
            .insert(name.to_string(), self.fields.len() as u32 - 1);
    }

    /// (dim, metric) of a vector field; None = not a vector field
    pub fn vector_field(&self, id: u32) -> Option<(u32, Metric)> {
        match self.fields.get(id as usize).map(|f| f.kind) {
            Some(FieldKind::Vector { dim, metric }) => Some((dim, metric)),
            _ => None,
        }
    }

    /// (field id, dim, metric) of the vector field named `name`
    pub fn vector_field_by_name(&self, name: &str) -> Option<(u32, u32, Metric)> {
        let id = self.field_id(name)?;
        let (dim, metric) = self.vector_field(id)?;
        Some((id, dim, metric))
    }

    /// ids of every vector field, ascending
    pub fn vector_field_ids(&self) -> Vec<u32> {
        self.fields
            .iter()
            .enumerate()
            .filter(|(_, f)| matches!(f.kind, FieldKind::Vector { .. }))
            .map(|(i, _)| i as u32)
            .collect()
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

/// one document: optional external id, stored-only metadata, field -> text,
/// field -> embedding. Unknown fields are rejected at add time.
#[derive(Debug, Clone, Default)]
pub struct Document {
    /// user-assigned identity (unique across the live index); None = implicit internal id
    pub id: Option<String>,
    /// arbitrary JSON, stored only (never tokenized or scored)
    pub metadata: Option<serde_json::Value>,
    /// text fields (tokenized, scored, stored)
    pub fields: HashMap<String, String>,
    /// vector fields: field name -> embedding. Values are stored exactly as
    /// supplied (f32); the caller owns the embedding model and the precision.
    pub vectors: HashMap<String, Vec<f32>>,
}

impl serde::Serialize for Document {
    /// wire shape: {"id": ..., "metadata": ..., "<field>": "<text>", "<vec>": [..], ...}
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
        for (k, v) in &self.vectors {
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
            // wire rule: a string is a text field, a number array is an
            // embedding. The SCHEMA decides which is which (a text field fed
            // an array is rejected at add time).
            match val {
                serde_json::Value::String(text) => {
                    doc.fields.insert(k.clone(), text.clone());
                }
                serde_json::Value::Array(items) => {
                    let mut vec = Vec::with_capacity(items.len());
                    for it in items {
                        let f = it.as_f64().ok_or_else(|| {
                            D::Error::custom(format!("`{k}` must contain only numbers"))
                        })?;
                        vec.push(f as f32);
                    }
                    doc.vectors.insert(k.clone(), vec);
                }
                _ => {
                    return Err(D::Error::custom(format!(
                        "`{k}` must be a string or an array of numbers"
                    )));
                }
            }
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

    /// attach an embedding to a vector field
    pub fn vector(mut self, field: &str, value: impl Into<Vec<f32>>) -> Self {
        self.vectors.insert(field.to_string(), value.into());
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
