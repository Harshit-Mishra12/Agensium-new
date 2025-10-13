# Master My Data Workflow Diagram

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Master My Data Tool                          │
│                    (Orchestrator v1.1.0)                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────────┐
        │  Input: CSV/Excel File + Parameters         │
        │  - source_system: "CRM_Data"                │
        │  - key_column: "email" (optional)           │
        └─────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        AGENT PIPELINE                            │
└─────────────────────────────────────────────────────────────────┘

    ┌──────────────────────────────────────────────────┐
    │  STEP 1: Source Tagger (v1.1.4)                  │
    │  ─────────────────────────────────────────────   │
    │  Input:  Raw file bytes                          │
    │  Action: Add _source_system column               │
    │  Output: Tagged data + base64 file               │
    │  Routing: → Entity Resolver                      │
    └──────────────────────────────────────────────────┘
                      │
                      │ [File passes with _source_system column]
                      ▼
    ┌──────────────────────────────────────────────────┐
    │  STEP 2: Entity Resolver (v1.1.0)                │
    │  ─────────────────────────────────────────────   │
    │  Input:  Tagged file + key_column                │
    │  Action: Find duplicate entities                 │
    │  Output: Duplicate analysis + base64 file        │
    │  Routing: → Dedup Agent or Clean Tool           │
    └──────────────────────────────────────────────────┘
                      │
                      │ [File passes unchanged]
                      ▼
    ┌──────────────────────────────────────────────────┐
    │  STEP 3: Deduplication Agent (v1.1.0)            │
    │  ─────────────────────────────────────────────   │
    │  Input:  Tagged file                             │
    │  Action: Detect exact row duplicates             │
    │  Output: Duplicate stats + base64 file           │
    │  Routing: → Compliance Tagger or Clean Tool      │
    └──────────────────────────────────────────────────┘
                      │
                      │ [File passes unchanged]
                      ▼
    ┌──────────────────────────────────────────────────┐
    │  STEP 4: Compliance Tagger (v1.1.0)              │
    │  ─────────────────────────────────────────────   │
    │  Input:  Tagged file                             │
    │  Action: Scan for PII fields                     │
    │  Output: PII flags + base64 file                 │
    │  Routing: → Governance Tool or Complete          │
    └──────────────────────────────────────────────────┘
                      │
                      ▼
        ┌─────────────────────────────────────┐
        │     Comprehensive Report             │
        │  ─────────────────────────────────   │
        │  • Source tagging summary            │
        │  • Entity resolution results         │
        │  • Deduplication statistics          │
        │  • Compliance/PII findings           │
        │  • All alerts aggregated             │
        │  • Final routing recommendation      │
        │  • Updated file (base64)             │
        └─────────────────────────────────────┘
```

## Data Flow Detail

```
┌─────────────────┐
│  Original File  │
│  customer.csv   │
│                 │
│  id,email,name  │
│  1,a@x.com,John │
│  2,b@x.com,Jane │
└─────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│  After Source Tagger            │
│                                 │
│  id,email,name,_source_system   │
│  1,a@x.com,John,CRM_Data        │
│  2,b@x.com,Jane,CRM_Data        │
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│  Entity Resolver Analysis       │
│                                 │
│  Key Column: email              │
│  Duplicates Found: 0            │
│  Status: ✓ Ready                │
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│  Dedup Agent Analysis           │
│                                 │
│  Total Rows: 2                  │
│  Duplicate Rows: 0              │
│  Status: ✓ Ready                │
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│  Compliance Tagger Analysis     │
│                                 │
│  PII Fields Found:              │
│  • email → Email                │
│  • name → Name                  │
│  Status: ⚠ Needs Review         │
└─────────────────────────────────┘
```

## Agent Communication Pattern

```
┌──────────────┐
│ Orchestrator │
└──────────────┘
       │
       │ 1. Prepare inputs (decode base64 if needed)
       ▼
┌──────────────┐
│    Agent     │◄──── file_contents (bytes)
│              │◄──── filename (str)
│              │◄──── params (dict)
└──────────────┘
       │
       │ 2. Process data
       │    • Read CSV/Excel
       │    • Process each sheet
       │    • Generate results
       │
       ▼
┌──────────────┐
│   Response   │
│              │
│ • results    │──── Per-sheet analysis
│ • audit      │──── Metadata
│ • updated_   │──── Base64-encoded file
│   file       │
└──────────────┘
       │
       │ 3. Extract updated_file.contents_b64
       ▼
┌──────────────┐
│ Orchestrator │──── Pass to next agent
└──────────────┘
```

## Multi-Sheet Processing

```
Excel File: sales_data.xlsx
├── Sheet1: Q1_Sales
├── Sheet2: Q2_Sales
└── Sheet3: Q3_Sales

Each Agent Processes All Sheets:

┌─────────────────────────────────────────┐
│  Agent Result Structure                 │
│                                         │
│  {                                      │
│    "results": {                         │
│      "Q1_Sales": {                      │
│        "status": "success",             │
│        "metadata": {...},               │
│        "alerts": [...],                 │
│        "routing": {...},                │
│        "data": {...}                    │
│      },                                 │
│      "Q2_Sales": {...},                 │
│      "Q3_Sales": {...}                  │
│    },                                   │
│    "updated_file": {                    │
│      "contents_b64": "...",             │
│      "filename": "sales_data.xlsx"      │
│    }                                    │
│  }                                      │
└─────────────────────────────────────────┘
```

## Alert Aggregation

```
┌──────────────────────────────────────────────────┐
│  Alert Collection Across All Agents              │
└──────────────────────────────────────────────────┘

Source Tagger:
  ✓ [INFO] Successfully tagged 1000 rows

Entity Resolver:
  ⚠ [WARNING] Found 5 duplicate entities

Dedup Agent:
  ⚠ [WARNING] Detected 10 duplicate rows (1%)

Compliance Tagger:
  ⚠ [CRITICAL] Found 3 PII fields

                    ↓
        ┌───────────────────────┐
        │  Aggregated Alerts    │
        │  (4 total alerts)     │
        └───────────────────────┘
```

## Routing Decision Flow

```
                    START
                      │
                      ▼
        ┌─────────────────────────┐
        │  All Agents Complete    │
        └─────────────────────────┘
                      │
                      ▼
        ┌─────────────────────────┐
        │  Check Final Routing    │
        │  (from last agent)      │
        └─────────────────────────┘
                      │
         ┌────────────┴────────────┐
         ▼                         ▼
    ┌─────────┐              ┌─────────┐
    │  Ready  │              │ Needs   │
    │         │              │ Review  │
    └─────────┘              └─────────┘
         │                         │
         ▼                         ▼
    No further            Suggest next tool:
    action needed         • Clean Data Tool
                         • Governance Tool
                         • Master Data Tool
```

## Error Handling

```
┌──────────────────────────────────────────┐
│  Error Handling at Each Level            │
└──────────────────────────────────────────┘

Agent Level:
  • Validate inputs
  • Check file format
  • Handle empty sheets
  • Catch exceptions
  • Return HTTPException

Orchestrator Level:
  • Validate workflow definition
  • Check agent registry
  • Handle missing parameters
  • Aggregate errors
  • Return comprehensive error

API Level:
  • Validate file upload
  • Check required parameters
  • Handle form data
  • Return HTTP 400/500
```

## Performance Considerations

```
┌──────────────────────────────────────────┐
│  Execution Time Breakdown (typical)      │
└──────────────────────────────────────────┘

Source Tagger:       0.5s  ████████
Entity Resolver:     0.8s  █████████████
Dedup Agent:         0.6s  ██████████
Compliance Tagger:   0.2s  ███

Total Workflow:      ~2.1s
Orchestration:       ~0.1s
────────────────────────────
Grand Total:         ~2.2s

Note: Times vary based on:
• File size
• Number of sheets
• Number of columns
• Number of rows
```
