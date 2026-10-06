from typing import List, Optional
from pydantic import BaseModel, Field

class LegalCrossReference(BaseModel):
    target_clause_or_act: str = Field(description="External clause/act referenced (e.g., 'Section 45 of Insurance Act', 'Ombudsman Rules 2017')")
    relationship_type: str = Field(description="One of: 'SUBJECT_TO', 'DERIVED_FROM', 'OVERRIDES', 'CLARIFIES'")

class NodeThreshold(BaseModel):
    name: str = Field(description="Normalized name, e.g., 'ped_waiting_period_cap'")
    metric: str = Field(description="'TIME', 'MONETARY', or 'CONDITIONAL'")
    numeric_value: Optional[float] = Field(default=None, description="Numerical value (null if qualitative or conditional)")
    unit: str = Field(description="'MONTHS', 'HOURS', 'DAYS', 'INR', or 'EVENT'")
    raw_expression: str = Field(description="Exact statutory phrase, e.g., 'within 36 months', '1 hour', 'immediately'")
    comparator: str = Field(description="'MAX', 'MIN', 'EXACT', 'WITHIN'")

class NodeRule(BaseModel):
    clause_identifier: str = Field(description="Exact statutory numbering, e.g., 'Clause 14(a)' or 'Section 45(2)'")
    title: str = Field(description="Short functional title of the rule")
    summary: str = Field(description="Clear summary of the legal mandate")
    verbatim_text: str = Field(description="Exact quote from statutory text")
    domain: str = Field(description="'HEALTH', 'MOTOR', 'LIFE', 'GENERAL', or 'UNIVERSAL'")
    thresholds: List[NodeThreshold] = Field(default_factory=list)
    exceptions: List[str] = Field(default_factory=list, description="Verbatim list of 'Provided that', 'However', or 'Except' clauses")
    cross_references: List[LegalCrossReference] = Field(default_factory=list)

class LegalExtractionBatch(BaseModel):
    document_id: str
    rules: List[NodeRule]