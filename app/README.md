1️⃣ Schema Drift (Structural Changes)

Changes in the structure of the dataset.

Scenario Example CSV Detection Notes
New column added customers_Jan.csv → customers_June.csv adds ReferralCode Column exists in current but not baseline JSON: "schema_change": "new column detected: ReferralCode"
Column removed products_Q1.csv → products_Q2.csv missing Discount column Column exists in baseline but not current JSON: "schema_change": "column missing: Discount"
Column renamed customers_Jan.csv changes PhoneNumber → ContactNumber Detected as removed + new column Optional fuzzy matching to detect renames
Data type changed orders_Jan.csv column OrderID int → string Compare dtypes of shared columns JSON: "schema_change": "type change: OrderID int→string"
2️⃣ Numeric Data Drift

Distribution changes in numeric columns.

Scenario Example CSV Detection Notes
Mean shift customers_Age column: mean 35 → 42 Compare mean values "drift_score": 0.78, "direction": "increase in mean age"
Std deviation change orders_Amount std 20 → 50 Standard deviation comparison Shows variability change
Outlier introduction products_Price suddenly has 10 extreme values IQR or z-score comparison JSON reports drift score + outliers
KS test distribution drift orders_Quantity distributions differ significantly Kolmogorov-Smirnov test "drift_score": 0.65, "p_value": 0.02"
3️⃣ Categorical Data Drift

Changes in categorical columns.

Scenario Example CSV Detection Notes
New categories appear customers_Country: adds South Korea Compare unique categories "direction": "new categories appeared"
Categories removed products_Category: Electronics missing Compare unique categories "direction": "category removed"
Distribution shift orders_Status: 80% shipped → 50% shipped Chi-square test "drift_score": 0.45, "p_value": 0.05"
Entropy change customers_Segment: more even distribution Entropy calculation Shows increased unpredictability
4️⃣ Missing Value Drift

Changes in percentage of missing values.

Scenario Example CSV Detection Notes
Increase in missing values customers_Email 2% → 10% missing Compare missing % "missing_values": {"Email": 10}
Decrease in missing values products_Stock 15% → 0% missing Compare missing % "missing_values": {"Stock": 0}
5️⃣ Temporal / Date Drift

Changes in date-related fields.

Scenario Example CSV Detection Notes
New date range orders_OrderDate: previous max 2025-03-31 → current max 2025-06-30 Compare min/max dates "date_range_shift": "new latest date"
Gap in dates Missing months in current dataset Compare date continuity "note": "missing data for April-May"
Change in event frequency events_LoginDate daily log counts differ Compare counts per period "drift_score": 0.5, "direction": "decrease in logins"
6️⃣ Multi-Scenario Drift

Some columns may have combined drifts.

Scenario Example CSV Detection Notes
Numeric + Missing drift customers_Age: mean shifted + 5% more missing Combine numeric drift + missing JSON combines "drift_score" and "missing_values"
Categorical + Schema drift customers_Country: new category + column renamed Combined schema + categorical drift "schema_change" + "drift_score"
