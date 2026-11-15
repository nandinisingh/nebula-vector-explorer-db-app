Implementation Plan: Embedding Exploration Tool for Recommendation Models
1. Current State Assessment
Findings:
No existing project files, code, or data detected
No uploaded embeddings, metadata, or evaluation results yet
Starting from scratch with a clean slate
No existing privacy or aggregation constraints identified
Confirmed Requirements:
Embedding dimensionality: 256
User/item embeddings: Same space (cosine similarity)
Similarity metric: Cosine
Projections: Generate synthetic 2D UMAP projections
Metadata fields: Genre, release era, popularity, audio features (all synthetic)
Evaluation metrics: NDCG@50, Precision@50, Recall@50
Per-user results: Yes, synthetic data at per-user level
Dataset size: 1,000 users, 100,000 items (70% music, 30% podcasts)
Framework: Streamlit
Deployment: Local machine
Data updates: Static (no real-time updates)

2. Proposed User Experience & Views
Primary View: Interactive Embedding Space Explorer
A single-page application with three coordinated panels:
Panel A: 2D Embedding Projection (Dominant View)
Visualization: Scatter plot of items in 2D space (UMAP projection from 256-dim embeddings)
Visual encoding:
Points represent items
Color encodes item type (music vs. podcast) by default
Size encodes popularity (if available)
Shape could distinguish types if color is used for another attribute
Interactions:
Click an item → highlights it and shows nearest neighbors
Hover → tooltip with item name, type, key metadata
Brush/lasso selection → select multiple items for batch inspection
Controls (sidebar):
View mode: Toggle between "Items only", "Users only", "Both"
Color by: Dropdown (item type, genre, release era, popularity quartile, audio feature bins)
Filter by: Checkboxes for item type, sliders for numeric attributes
Show user: Text input or dropdown to select a user ID → plots user embedding as a distinct marker (star/diamond)
Side-by-side comparison: Select two items to compare
Panel B: Nearest Neighbors / Similarity Rankings
Triggered by: Selecting an item or entering a user ID
Item–item mode (item selected):
Table showing top-20 nearest items by embedding similarity
Columns: rank, item name, type, genre, similarity score, distance
Clicking a neighbor → selects it in Panel A and updates this panel
User–item mode (user selected):
Table showing top-50 items ranked by user–item similarity (candidate generation ranking)
Columns: rank, item name, type, genre, similarity score
Clear label: "Items ranked by embedding similarity for User [ID]"
Note: "This ranking represents candidate generation, not final recommendations"
Side-by-side comparison mode (two items selected):
Shows both items' metadata
Displays similarity score between them
Shows their respective top-20 neighbors
Highlights common neighbors
Panel C: Evaluation Results Inspector
Default state: Shows aggregate metrics for Model 1
Visual: Bar chart showing NDCG@50, Precision@50, Recall@50
Table: metric name, value, description
User-specific mode (user selected):
Shows per-user evaluation results
Displays reference items (ground truth) vs. model's top-50 ranked items
Visual comparison: Venn diagram showing overlap between ground truth and predictions
Metrics: per-user NDCG@50, Precision@50, Recall@50 with comparison to aggregate
Item-specific mode (item selected):
Shows how often this item appears in top-50 for users who interacted with it (if data supports this)
Or: shows aggregate stats for items similar to the selected one
Secondary Elements
Header: Model selector (initially just "Model 1", grayed out; will become dropdown when Model 2 is added)
Footer/info panel:
Projection disclaimer: "2D projection preserves local structure but distorts global distances. Similarity scores are computed in the original 256-dimensional embedding space."
Legend for color/size encodings
Data summary: number of items, users, embedding dimensions

3. Required Data & Mapping to Views
Input Data (All Synthetic)
A. Embeddings
Item embeddings: 100,000 items × 256 dimensions
Format: CSV with columns item_id, dim_0, dim_1, ..., dim_255
Clustering: Items cluster by genre (music) or topic (podcasts)
Normalization: L2-normalized for cosine similarity
User embeddings: 1,000 users × 256 dimensions
Format: CSV with columns user_id, dim_0, dim_1, ..., dim_255
Positioning: Near preferred genres/topics
2D projections: Precomputed UMAP
Format: CSV with columns item_id, x, y, projection_method
B. Item Metadata
Required columns:
item_id (string: "music_000001" or "podcast_000001")
item_name (string: human-readable)
item_type (string: "music" or "podcast")
genre (string: music genres or podcast topics)
release_year (int: 1960-2024 for music, 2015-2024 for podcasts)
popularity_score (float: 0-1, beta distribution)
duration_seconds (int)
energy (float: 0-1, audio feature)
valence (float: 0-1, audio feature for happiness)
tempo (int: 60-200, music only)
acousticness (float: 0-1, music only)
C. User Metadata
Columns:
user_id (string: "user_0001")
user_segment (string: "casual", "regular", "power_user")
signup_year (int: 2018-2024)
primary_preference (string: "music" or "podcast")
D. Evaluation Results
Per-user results CSV:
user_id
ndcg_at_50 (float)
precision_at_50 (float)
recall_at_50 (float)
ground_truth_items (list of item IDs, 5-20 per user)
top_k_predicted_items (list of top-50 item IDs)
Aggregate metrics JSON:
model_id: "model_1"
metrics: array of {metric_name, metric_value, description}
Data Mapping to Views
Data
Panel A (Projection)
Panel B (Neighbors)
Panel C (Eval)
Item embeddings (2D)
Point positions
—
—
Item metadata
Color, size, tooltip
Table columns
Item names in results
Item embeddings (256D)
—
Similarity calculation
—
User embeddings (256D)
User marker position
User–item similarity
—
User metadata
Tooltip for user marker
—
User context
Evaluation results
—
—
Metrics charts, ground truth vs. predicted

4. Files & Components to Create
Project Structure
embedding_explorer/
├── README.md                          # Project overview, setup, usage
├── requirements.txt                   # Python dependencies
├── generate_synthetic_data.py         # Data generation script (run once)
├── data/                              # Data directory
│   └── model_1/
│       ├── item_embeddings.csv
│       ├── user_embeddings.csv
│       ├── item_metadata.csv
│       ├── user_metadata.csv
│       ├── item_embeddings_2d.csv
│       ├── evaluation_results_per_user.csv
│       └── evaluation_metrics_aggregate.json
└── src/
    ├── app.py                         # Main Streamlit app
    ├── data_loader.py                 # Load and validate input data
    ├── similarity.py                  # Compute nearest neighbors, user–item rankings
    └── components/
        ├── embedding_plot.py          # Panel A: scatter plot component
        ├── neighbor_table.py          # Panel B: similarity table component
        ├── eval_panel.py              # Panel C: evaluation results component
        └── comparison_view.py         # Side-by-side item comparison
Key Components
generate_synthetic_data.py (Already created in Canvas 1)
Generates 100K items with 256-dim embeddings clustered by genre/topic
Generates 1K users with embeddings near their preferences
Computes UMAP 2D projections
Creates metadata with realistic distributions
Generates evaluation results (per-user and aggregate)
data_loader.py
Functions:
load_embeddings(model_id) → returns DataFrames for user and item embeddings
load_metadata(model_id) → returns item and user metadata DataFrames
load_2d_projections(model_id) → returns 2D coordinates
load_evaluation_results(model_id) → returns evaluation metrics dict and per-user DataFrame
validate_data(...) → checks for required columns, matching IDs, etc.
Error handling: clear messages if files are missing or malformed
similarity.py
Functions:
compute_item_neighbors(item_id, embeddings, top_k=20) → list of (neighbor_id, similarity_score)
compute_user_item_ranking(user_id, user_embeddings, item_embeddings, top_k=50) → ranked list of (item_id, similarity_score)
Uses cosine similarity via sklearn.metrics.pairwise.cosine_similarity
app.py (Streamlit-based)
Layout:
Sidebar: model selector, view mode toggle, color/filter controls, user/item input, comparison controls
Main area: three-panel layout (Panel A dominant, Panels B & C below)
State management: Streamlit session state to track selected item/user, comparison items
Callbacks: update plots and tables on selection
components/embedding_plot.py
Plotly scattergl (WebGL for 100K points) with:
Hover data: item name, type, metadata
Click event → updates session state with selected item
Color scale based on selected attribute
User markers (if view mode includes users)
Comparison highlights (if two items selected)
components/neighbor_table.py
Displays similarity rankings as an interactive table
Clickable rows → select that item in the plot
Supports item–item, user–item, and comparison modes
components/eval_panel.py
Displays aggregate metrics (bar chart + table)
Displays per-user metrics and ground truth vs. predicted items (when user is selected)
Visual: Venn diagram for item overlap, bar chart comparing per-user vs. aggregate metrics
components/comparison_view.py
Side-by-side display of two selected items
Shows metadata, similarity score between them
Shows their respective neighbors with common neighbors highlighted

5. Key Technical Choices, Risks & Assumptions
Technical Choices
Decision
Choice
Rationale
Framework
Streamlit
Rapid prototyping, Python-native, easy to iterate
Projection
UMAP (precomputed)
Preserves local and some global structure, faster than t-SNE
Similarity metric
Cosine similarity
Standard for embeddings, handles different magnitudes
Data format
CSV for embeddings/metadata, JSON for eval results
Simple, human-readable, easy to inspect
Plotting
Plotly scattergl (WebGL)
Handles 100K points smoothly
Caching
Streamlit @st.cache_data
Avoid recomputing on every interaction
Risks & Mitigations
Risk
Impact
Mitigation
100K points slow to render
Initial load lag
Use Plotly's scattergl (WebGL); show loading spinner
Similarity computation slow
Interaction lag
Precompute and cache all similarities on load; use numpy vectorization
Large datasets use too much memory
App crashes
Load embeddings as float32 (not float64); subsample for initial exploration if needed
Projection distorts distances
Users misinterpret 2D proximity
Prominent disclaimer; always show similarity scores from 256D space in Panel B
Assumptions
Embeddings are dense vectors of 256 dimensions, L2-normalized
User and item embeddings are in the same space, allowing direct cosine similarity
Evaluation results are supplied — we do NOT compute NDCG or other metrics from scratch
Ground truth data is included in evaluation results for per-user metrics
Item IDs are consistent across all files
Model 1 is the only model initially; design supports adding Model 2 later

6. Acceptance Criteria & Verification
Acceptance Criteria
Functional Requirements
✅ Load Model 1 data: Successfully loads all files from data/model_1/
✅ Display 2D projection: Shows 100K items as a scatter plot, colored by item type
✅ Item selection: Clicking an item highlights it and displays its top-20 nearest neighbors
✅ User–item ranking: Entering a user ID shows top-50 items ranked by similarity
✅ Evaluation results: Panel C displays aggregate metrics by default
✅ Per-user evaluation: Selecting a user shows per-user metrics and ground truth vs. predicted items
✅ Attribute exploration: Changing "Color by" dropdown re-colors the plot
✅ Filtering: Applying filters updates the plot to show only matching items
✅ View modes: Toggle between items only, users only, or both
✅ Side-by-side comparison: Select two items to see their similarity and common neighbors
✅ Projection disclaimer: Visible text explains that 2D distances are approximate
Non-Functional Requirements
✅ Performance: Plot renders in <30 seconds for 100K items (with WebGL)
✅ Extensibility: Adding Model 2 requires only adding data/model_2/ directory
✅ Error handling: Clear error messages if data files are missing or malformed
✅ Documentation: README includes setup instructions and usage examples
Verification Plan
Phase 1: Data Generation
Run generate_synthetic_data.py
Verify 7 files created in data/model_1/
Check file sizes and row counts
Inspect genre/topic distributions
Phase 2: Core Logic Testing
Load data with data_loader.py → verify no errors
Compute item neighbors → verify top-20 list is sorted by similarity
Compute user–item ranking → verify top-50 list is sorted by similarity
Phase 3: UI Testing
Load app → verify plot shows 100K points
Click an item → verify Panel B updates with neighbors
Enter a user ID → verify Panel B shows user–item ranking
Change "Color by" to genre → verify colors update
Apply filter (podcasts only) → verify plot shows 30K points
Select two items → verify comparison view appears
Toggle view mode → verify users appear/disappear
Phase 4: Performance Testing
Measure initial load time (should be <30 seconds)
Measure interaction latency (click → update, should be <1 second)

7. Answered Questions
Data & Model Questions
✅ Dimensionality: 256
✅ Same space: Yes
✅ Similarity metric: Cosine
✅ Projections: Generate synthetic UMAP
✅ Metadata fields: Genre, release era, popularity, audio features (all synthetic)
✅ Evaluation metrics: NDCG@50, Precision@50, Recall@50
✅ Per-user results: Yes, synthetic
✅ Dataset size: 1,000 users, 100,000 items (70% music, 30% podcasts)
Product & Scope Questions
✅ Export: No
✅ Upload UI: No (developer-facing, data in data/ directory)
✅ Privacy: No constraints
✅ Side-by-side comparison: Yes
✅ User visualization: Both (separate view and combined view)
✅ Model 2 comparison: Side-by-side or overlay, difference view, metric comparison (visual, not tables)
Technical Questions
✅ Framework: Streamlit
✅ Deployment: Local machine
✅ Real-time updates: No (static analysis tool)

8. Implementation Phases
Phase 1: Generate Synthetic Data ✅ Ready
Run generate_synthetic_data.py
Output: 7 files in data/model_1/
Time: ~5-10 minutes
Phase 2: Build Core Modules (Next)
Create data_loader.py
Create similarity.py
Test with synthetic data
Phase 3: Build Streamlit App (After Phase 2)
Create app.py
Create all components
Test all interactions
Phase 4: Documentation & Polish (Final)
Create README.md
Create requirements.txt
Final testing and verification
Phase 5: Prepare for Model 2 (Future)
Structure already supports adding data/model_2/
Model selector dropdown
Comparison views

Summary
This plan proposes a three-panel exploratory tool for Model 1 embeddings:
Panel A: 2D scatter plot of 100K items (dominant view), with controls for coloring, filtering, and view modes
Panel B: Nearest neighbors (item–item), similarity rankings (user–item), or side-by-side comparison
Panel C: Evaluation results (aggregate and per-user) with visual comparisons
The design is:
✅ Extensible (Model 2 can be added with minimal changes)
✅ Exploratory (no predetermined narrative)
✅ Grounded in supplied data (no evaluation metrics are computed)
✅ Performant (WebGL for 100K points, caching for interactions)
✅ Autonomous (synthetic data generation is fully automated)
Current Status: Phase 1 (data generation script) is ready. Awaiting execution to proceed to Phase 2.
