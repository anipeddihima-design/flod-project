import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import random

from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist


# ============================================================
# STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Krishna-Godavari Flood System",
    page_icon="🌊",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🌊 Krishna-Godavari Flood Forecasting & Disaster Response")

st.markdown(
    """
    ### AI + Quantum Optimization Based Flood Management System

    This system combines:

    - 🌧️ Flood forecasting
    - 🤖 Machine Learning
    - 📍 Sensor placement optimization
    - 🧬 Genetic Algorithm
    - ⚛️ Quantum/QAOA optimization
    - 🚨 Disaster response
    """
)


# ============================================================
# FILE PATHS
# ============================================================

DATA_FILE = "flood_model_dataset.csv"

RF_MODEL_FILE = "flood_risk_random_forest.pkl"

XGB_MODEL_FILE = "flood_risk_xgboost.pkl"


# ============================================================
# CHECK FILES
# ============================================================

required_files = [
    DATA_FILE,
    RF_MODEL_FILE,
    XGB_MODEL_FILE
]

missing_files = [
    file for file in required_files
    if not os.path.exists(file)
]

if missing_files:

    st.error("❌ Required files are missing.")

    st.write("Missing files:")

    for file in missing_files:
        st.write(f"- `{file}`")

    st.stop()


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    return pd.read_csv(DATA_FILE)


@st.cache_resource
def load_models():

    rf = joblib.load(RF_MODEL_FILE)

    xgb = joblib.load(XGB_MODEL_FILE)

    return rf, xgb


try:

    df = load_data()

    rf_model, xgb_model = load_models()

except Exception as e:

    st.error(
        f"Error loading project files: {e}"
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🌊 Navigation")

page = st.sidebar.radio(
    "Select Module",
    [
        "🏠 Dashboard",
        "🌧️ Flood Prediction",
        "📊 Flood Risk Analysis",
        "📍 Sensor Placement",
        "⚛️ Quantum Optimization",
        "🚨 Disaster Response"
    ]
)


# ============================================================
# FEATURES
# ============================================================

features = [
    "latitude",
    "longitude",
    "rainfall_1h",
    "rainfall_3h",
    "rainfall_6h",
    "rainfall_12h",
    "rainfall_24h",
    "rainfall_72h",
    "river_level",
    "river_level_lag1",
    "river_level_lag3",
    "river_level_lag6",
    "river_level_lag12",
    "river_discharge",
    "temperature",
    "humidity",
    "pressure",
    "wind_speed",
    "elevation",
    "distance_from_river",
    "historical_flood_count",
    "population"
]


# ============================================================
# CHECK FEATURES
# ============================================================

missing_features = [
    feature
    for feature in features
    if feature not in df.columns
]

if missing_features:

    st.error(
        "The following Phase 1 features are missing from "
        "your dataset:"
    )

    st.write(missing_features)

    st.info(
        "Make sure your flood_model_dataset.csv contains "
        "the same features used during Phase 1."
    )

    st.stop()


# ============================================================
# FLOOD PROBABILITY FUNCTION
# ============================================================

def get_flood_probability(model, X):

    try:

        probabilities = model.predict_proba(X)

        # Binary classification
        if probabilities.shape[1] == 2:

            return probabilities[:, 1]

        # Multi-class classification
        elif probabilities.shape[1] >= 4:

            probability = (
                probabilities[:, 0] * 0.05 +
                probabilities[:, 1] * 0.30 +
                probabilities[:, 2] * 0.70 +
                probabilities[:, 3] * 0.95
            )

            return probability

        else:

            return probabilities.max(axis=1)

    except Exception:

        predictions = model.predict(X)

        predictions = np.asarray(
            predictions,
            dtype=float
        )

        if predictions.max() > predictions.min():

            predictions = (
                predictions -
                predictions.min()
            ) / (
                predictions.max() -
                predictions.min()
            )

        return predictions


# ============================================================
# GENERATE FLOOD PREDICTIONS
# ============================================================

@st.cache_data
def generate_predictions(data):

    X = data[features]

    rf_probability = get_flood_probability(
        rf_model,
        X
    )

    xgb_probability = get_flood_probability(
        xgb_model,
        X
    )

    result = data.copy()

    result["rf_flood_probability"] = rf_probability

    result["xgb_flood_probability"] = xgb_probability

    result["flood_probability"] = (
        rf_probability +
        xgb_probability
    ) / 2

    result["flood_probability"] = (
        result["flood_probability"]
        .clip(0, 1)
    )

    return result


try:

    prediction_df = generate_predictions(df)

except Exception as e:

    st.error(
        f"Error generating flood predictions: {e}"
    )

    st.stop()


# ============================================================
# RISK CATEGORY
# ============================================================

def risk_category(probability):

    if probability < 0.25:

        return "LOW"

    elif probability < 0.50:

        return "MODERATE"

    elif probability < 0.75:

        return "HIGH"

    else:

        return "VERY HIGH"


prediction_df["predicted_risk"] = (
    prediction_df["flood_probability"]
    .apply(risk_category)
)


# ============================================================
# SENSOR PRIORITY
# ============================================================

def calculate_priority(data):

    result = data.copy()

    risk_scaler = MinMaxScaler()

    result["risk_score"] = (
        risk_scaler.fit_transform(
            result[["flood_probability"]]
        )
        .ravel()
    )

    population_scaler = MinMaxScaler()

    result["population_score"] = (
        population_scaler.fit_transform(
            result[["population"]]
        )
        .ravel()
    )

    elevation_scaler = MinMaxScaler()

    result["elevation_score"] = (
        1 -
        elevation_scaler.fit_transform(
            result[["elevation"]]
        )
        .ravel()
    )

    river_scaler = MinMaxScaler()

    result["river_proximity_score"] = (
        1 -
        river_scaler.fit_transform(
            result[["distance_from_river"]]
        )
        .ravel()
    )

    result["priority_score"] = (

        0.45 *
        result["risk_score"]

        +

        0.25 *
        result["population_score"]

        +

        0.15 *
        result["elevation_score"]

        +

        0.15 *
        result["river_proximity_score"]
    )

    return result


prediction_df = calculate_priority(
    prediction_df
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.header(
        "🏠 Flood Forecasting Dashboard"
    )

    st.markdown(
        """
        This dashboard provides an integrated flood
        forecasting and disaster-response framework for
        the Krishna-Godavari basin region.
        """
    )

    col1, col2, col3, col4 = st.columns(4)

    total_locations = len(
        prediction_df
    )

    high_risk = len(
        prediction_df[
            prediction_df["flood_probability"] >= 0.50
        ]
    )

    very_high = len(
        prediction_df[
            prediction_df["flood_probability"] >= 0.75
        ]
    )

    average_risk = (
        prediction_df["flood_probability"]
        .mean()
    )

    col1.metric(
        "📍 Locations",
        total_locations
    )

    col2.metric(
        "⚠️ High Risk",
        high_risk
    )

    col3.metric(
        "🚨 Very High Risk",
        very_high
    )

    col4.metric(
        "🌊 Average Flood Probability",
        f"{average_risk * 100:.1f}%"
    )

    st.divider()

    st.subheader(
        "Flood Risk Distribution"
    )

    risk_counts = (
        prediction_df[
            "predicted_risk"
        ]
        .value_counts()
    )

    st.bar_chart(
        risk_counts
    )

    st.subheader(
        "Highest Risk Locations"
    )

    display_columns = [
        "latitude",
        "longitude",
        "flood_probability",
        "predicted_risk",
        "river_level",
        "rainfall_24h",
        "population"
    ]

    highest_risk = (
        prediction_df
        .sort_values(
            "flood_probability",
            ascending=False
        )
        .head(10)
    )

    st.dataframe(
        highest_risk[display_columns],
        use_container_width=True
    )


# ============================================================
# FLOOD PREDICTION
# ============================================================

elif page == "🌧️ Flood Prediction":

    st.header(
        "🌧️ Flood Prediction"
    )

    st.write(
        "Predictions generated using the Phase 1 "
        "Random Forest and XGBoost models."
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "Random Forest"
        )

        rf_average = (
            prediction_df[
                "rf_flood_probability"
            ].mean()
        )

        st.metric(
            "Average Flood Probability",
            f"{rf_average * 100:.2f}%"
        )

    with col2:

        st.subheader(
            "XGBoost"
        )

        xgb_average = (
            prediction_df[
                "xgb_flood_probability"
            ].mean()
        )

        st.metric(
            "Average Flood Probability",
            f"{xgb_average * 100:.2f}%"
        )

    st.divider()

    st.subheader(
        "Flood Prediction Results"
    )

    columns = [
        "latitude",
        "longitude",
        "rf_flood_probability",
        "xgb_flood_probability",
        "flood_probability",
        "predicted_risk"
    ]

    st.dataframe(
        prediction_df[columns],
        use_container_width=True
    )

    st.subheader(
        "Flood Probability Distribution"
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.hist(
        prediction_df[
            "flood_probability"
        ],
        bins=20
    )

    ax.set_xlabel(
        "Flood Probability"
    )

    ax.set_ylabel(
        "Number of Locations"
    )

    ax.set_title(
        "Predicted Flood Probability"
    )

    st.pyplot(fig)


# ============================================================
# FLOOD RISK ANALYSIS
# ============================================================

elif page == "📊 Flood Risk Analysis":

    st.header(
        "📊 Flood Risk Analysis"
    )

    risk_count = (
        prediction_df[
            "predicted_risk"
        ]
        .value_counts()
    )

    st.subheader(
        "Risk Categories"
    )

    st.dataframe(
        risk_count.rename(
            "Number of Locations"
        ),
        use_container_width=True
    )

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    risk_count.plot(
        kind="bar",
        ax=ax
    )

    ax.set_xlabel(
        "Flood Risk"
    )

    ax.set_ylabel(
        "Number of Locations"
    )

    ax.set_title(
        "Flood Risk Distribution"
    )

    plt.xticks(
        rotation=0
    )

    st.pyplot(fig)

    st.subheader(
        "Top 20 Critical Locations"
    )

    critical = (
        prediction_df
        .sort_values(
            "flood_probability",
            ascending=False
        )
        .head(20)
    )

    st.dataframe(
        critical[
            [
                "latitude",
                "longitude",
                "flood_probability",
                "predicted_risk",
                "rainfall_24h",
                "river_level",
                "population",
                "elevation"
            ]
        ],
        use_container_width=True
    )


# ============================================================
# SENSOR PLACEMENT
# ============================================================

elif page == "📍 Sensor Placement":

    st.header(
        "📍 Disaster Sensor Placement"
    )

    st.write(
        """
        Sensor locations are selected using flood risk,
        population, elevation and river proximity.
        """
    )

    number_of_sensors = st.slider(
        "Number of sensors",
        min_value=3,
        max_value=min(
            20,
            len(prediction_df)
        ),
        value=min(
            10,
            len(prediction_df)
        )
    )

    candidate_count = st.slider(
        "Number of candidate locations",
        min_value=max(
            number_of_sensors,
            10
        ),
        max_value=min(
            50,
            len(prediction_df)
        ),
        value=min(
            20,
            len(prediction_df)
        )
    )

    candidates = (
        prediction_df
        .sort_values(
            "priority_score",
            ascending=False
        )
        .head(candidate_count)
        .copy()
        .reset_index(drop=True)
    )

    candidates[
        "location_id"
    ] = [
        f"L{i+1:03d}"
        for i in range(
            len(candidates)
        )
    ]

    st.subheader(
        "Candidate Sensor Locations"
    )

    st.dataframe(
        candidates[
            [
                "location_id",
                "latitude",
                "longitude",
                "flood_probability",
                "predicted_risk",
                "population",
                "priority_score"
            ]
        ],
        use_container_width=True
    )

    coordinates = candidates[
        [
            "latitude",
            "longitude"
        ]
    ].values

    distance_matrix = cdist(
        coordinates,
        coordinates
    )

    coverage_distance = 0.08

    coverage_matrix = (
        distance_matrix <= coverage_distance
    ).astype(int)

    priorities = (
        candidates[
            "priority_score"
        ]
        .values
    )

    # --------------------------------------------------------
    # GREEDY
    # --------------------------------------------------------

    def greedy_algorithm():

        selected = []

        covered = np.zeros(
            len(candidates),
            dtype=int
        )

        for _ in range(
            number_of_sensors
        ):

            best_location = None

            best_score = -999999

            for i in range(
                len(candidates)
            ):

                if i in selected:
                    continue

                new_coverage = (
                    coverage_matrix[i] *
                    (1 - covered)
                )

                score = np.sum(
                    new_coverage *
                    priorities
                )

                if score > best_score:

                    best_score = score

                    best_location = i

            if best_location is not None:

                selected.append(
                    best_location
                )

                covered = np.maximum(
                    covered,
                    coverage_matrix[
                        best_location
                    ]
                )

        return selected

    greedy_selected = (
        greedy_algorithm()
    )

    # --------------------------------------------------------
    # GENETIC ALGORITHM
    # --------------------------------------------------------

    def create_chromosome():

        chromosome = np.zeros(
            len(candidates),
            dtype=int
        )

        selected = np.random.choice(
            len(candidates),
            number_of_sensors,
            replace=False
        )

        chromosome[
            selected
        ] = 1

        return chromosome

    def chromosome_fitness(
        chromosome
    ):

        selected = np.where(
            chromosome == 1
        )[0]

        if len(selected) != number_of_sensors:

            return -999999

        covered = np.zeros(
            len(candidates)
        )

        for location in selected:

            covered = np.maximum(
                covered,
                coverage_matrix[
                    location
                ]
            )

        return np.sum(
            covered *
            priorities
        )

    def repair(
        chromosome
    ):

        chromosome = chromosome.copy()

        while np.sum(
            chromosome
        ) > number_of_sensors:

            selected = np.where(
                chromosome == 1
            )[0]

            index = np.random.choice(
                selected
            )

            chromosome[index] = 0

        while np.sum(
            chromosome
        ) < number_of_sensors:

            unselected = np.where(
                chromosome == 0
            )[0]

            index = np.random.choice(
                unselected
            )

            chromosome[index] = 1

        return chromosome

    def genetic_algorithm():

        population_size = 60

        generations = 80

        population = [
            create_chromosome()
            for _ in range(
                population_size
            )
        ]

        best = None

        best_score = -999999

        for generation in range(
            generations
        ):

            scores = [
                chromosome_fitness(
                    chromosome
                )
                for chromosome in population
            ]

            best_index = np.argmax(
                scores
            )

            if scores[
                best_index
            ] > best_score:

                best_score = scores[
                    best_index
                ]

                best = population[
                    best_index
                ].copy()

            elite_indices = np.argsort(
                scores
            )[-10:]

            new_population = [
                population[index].copy()
                for index in elite_indices
            ]

            while len(
                new_population
            ) < population_size:

                parent1 = population[
                    random.choice(
                        elite_indices
                    )
                ]

                parent2 = population[
                    random.choice(
                        elite_indices
                    )
                ]

                point = np.random.randint(
                    1,
                    len(candidates) - 1
                )

                child = np.concatenate(
                    [
                        parent1[:point],
                        parent2[point:]
                    ]
                )

                if np.random.random() < 0.15:

                    mutation = np.random.randint(
                        len(candidates)
                    )

                    child[
                        mutation
                    ] = 1 - child[
                        mutation
                    ]

                child = repair(
                    child
                )

                new_population.append(
                    child
                )

            population = new_population

        return np.where(
            best == 1
        )[0].tolist()

    genetic_selected = (
        genetic_algorithm()
    )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    greedy_locations = candidates.iloc[
        greedy_selected
    ]

    genetic_locations = candidates.iloc[
        genetic_selected
    ]

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "🟢 Greedy Algorithm"
        )

        st.dataframe(
            greedy_locations[
                [
                    "location_id",
                    "latitude",
                    "longitude",
                    "flood_probability",
                    "priority_score"
                ]
            ],
            use_container_width=True
        )

    with col2:

        st.subheader(
            "🧬 Genetic Algorithm"
        )

        st.dataframe(
            genetic_locations[
                [
                    "location_id",
                    "latitude",
                    "longitude",
                    "flood_probability",
                    "priority_score"
                ]
            ],
            use_container_width=True
        )

    # --------------------------------------------------------
    # MAP
    # --------------------------------------------------------

    st.subheader(
        "📍 Sensor Placement Map"
    )

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    ax.scatter(
        candidates["longitude"],
        candidates["latitude"],
        s=70,
        label="Candidate Locations"
    )

    ax.scatter(
        greedy_locations["longitude"],
        greedy_locations["latitude"],
        s=180,
        marker="o",
        label="Greedy Sensors"
    )

    ax.scatter(
        genetic_locations["longitude"],
        genetic_locations["latitude"],
        s=180,
        marker="^",
        label="Genetic Sensors"
    )

    ax.set_xlabel(
        "Longitude"
    )

    ax.set_ylabel(
        "Latitude"
    )

    ax.set_title(
        "Optimized Sensor Placement"
    )

    ax.legend()

    ax.grid(
        alpha=0.3
    )

    st.pyplot(fig)


# ============================================================
# QUANTUM OPTIMIZATION
# ============================================================

elif page == "⚛️ Quantum Optimization":

    st.header(
        "⚛️ Quantum Sensor Placement Optimization"
    )

    st.markdown(
        """
        ### Quantum approach

        The limited sensor placement problem is formulated
        as a binary optimization problem.

        Each candidate location has a binary variable:

        **x = 1 → place sensor**

        **x = 0 → do not place sensor**

        The optimization attempts to maximize flood-risk
        coverage while using a limited number of sensors.
        """
    )

    st.info(
        "QAOA is executed only when you press the button."
    )

    quantum_sensors = st.slider(
        "Number of quantum sensors",
        min_value=3,
        max_value=min(
            12,
            len(prediction_df)
        ),
        value=min(
            5,
            len(prediction_df)
        )
    )

    quantum_candidates = st.slider(
        "Quantum candidate locations",
        min_value=max(
            quantum_sensors,
            8
        ),
        max_value=min(
            16,
            len(prediction_df)
        ),
        value=min(
            10,
            len(prediction_df)
        )
    )

    quantum_df = (
        prediction_df
        .sort_values(
            "priority_score",
            ascending=False
        )
        .head(
            quantum_candidates
        )
        .copy()
        .reset_index(drop=True)
    )

    quantum_df[
        "location_id"
    ] = [
        f"Q{i+1:03d}"
        for i in range(
            len(quantum_df)
        )
    ]

    st.subheader(
        "Quantum Candidate Locations"
    )

    st.dataframe(
        quantum_df[
            [
                "location_id",
                "latitude",
                "longitude",
                "flood_probability",
                "predicted_risk",
                "priority_score"
            ]
        ],
        use_container_width=True
    )

    if st.button(
        "⚛️ Run Quantum Sensor Optimization",
        type="primary"
    ):

        with st.spinner(
            "Running quantum optimization..."
        ):

            coordinates = quantum_df[
                [
                    "latitude",
                    "longitude"
                ]
            ].values

            distance_matrix = cdist(
                coordinates,
                coordinates
            )

            quantum_coverage = (
                distance_matrix <= 0.08
            ).astype(int)

            priorities = quantum_df[
                "priority_score"
            ].values

            quantum_result = None

            # ------------------------------------------------
            # TRY QAOA
            # ------------------------------------------------

            try:

                from qiskit_optimization import (
                    QuadraticProgram
                )

                from qiskit_optimization.algorithms import (
                    MinimumEigenOptimizer
                )

                from qiskit_algorithms import (
                    QAOA
                )

                from qiskit_algorithms.optimizers import (
                    COBYLA
                )

                from qiskit_aer.primitives import (
                    SamplerV2
                )

                qp = QuadraticProgram(
                    "Flood_Sensor_Placement"
                )

                for i in range(
                    len(quantum_df)
                ):

                    qp.binary_var(
                        f"x_{i}"
                    )

                linear = {}

                for i in range(
                    len(quantum_df)
                ):

                    linear[
                        f"x_{i}"
                    ] = -float(
                        priorities[i]
                    )

                quadratic = {}

                for i in range(
                    len(quantum_df)
                ):

                    for j in range(
                        i + 1,
                        len(quantum_df)
                    ):

                        overlap = np.sum(
                            quantum_coverage[i] *
                            quantum_coverage[j]
                        )

                        if overlap > 0:

                            quadratic[
                                (
                                    f"x_{i}",
                                    f"x_{j}"
                                )
                            ] = (
                                0.01 *
                                float(overlap)
                            )

                qp.minimize(
                    linear=linear,
                    quadratic=quadratic
                )

                constraint = {
                    f"x_{i}": 1
                    for i in range(
                        len(quantum_df)
                    )
                }

                qp.linear_constraint(
                    linear=constraint,
                    sense="==",
                    rhs=quantum_sensors,
                    name="sensor_count"
                )

                sampler = SamplerV2()

                qaoa = QAOA(
                    sampler=sampler,
                    optimizer=COBYLA(
                        maxiter=50
                    ),
                    reps=1
                )

                optimizer = (
                    MinimumEigenOptimizer(
                        qaoa
                    )
                )

                result = optimizer.solve(
                    qp
                )

                quantum_result = np.where(
                    np.array(
                        [
                            result.variables[
                                i
                            ].value
                            for i in range(
                                len(quantum_df)
                            )
                        ]
                    ) > 0.5
                )[0]

                method = "QAOA"

            except Exception as error:

                # ------------------------------------------------
                # QUBO COMPATIBLE FALLBACK
                # ------------------------------------------------

                st.warning(
                    "QAOA could not be executed with the "
                    "installed Qiskit version. "
                    "Running the same binary optimization "
                    "problem using a classical QUBO search."
                )

                from itertools import combinations

                best_score = -999999

                best_combination = None

                for combination in combinations(
                    range(
                        len(quantum_df)
                    ),
                    quantum_sensors
                ):

                    covered = np.zeros(
                        len(quantum_df)
                    )

                    for location in combination:

                        covered = np.maximum(
                            covered,
                            quantum_coverage[
                                location
                            ]
                        )

                    score = np.sum(
                        covered *
                        priorities
                    )

                    if score > best_score:

                        best_score = score

                        best_combination = (
                            combination
                        )

                quantum_result = np.array(
                    best_combination
                )

                method = (
                    "QUBO-compatible classical fallback"
                )

            # ------------------------------------------------
            # DISPLAY RESULTS
            # ------------------------------------------------

            selected = quantum_df.iloc[
                quantum_result
            ].copy()

            st.success(
                f"Optimization completed using: {method}"
            )

            st.subheader(
                "⚛️ Selected Sensor Locations"
            )

            st.dataframe(
                selected[
                    [
                        "location_id",
                        "latitude",
                        "longitude",
                        "flood_probability",
                        "predicted_risk",
                        "population",
                        "priority_score"
                    ]
                ],
                use_container_width=True
            )

            # ------------------------------------------------
            # QUANTUM MAP
            # ------------------------------------------------

            fig, ax = plt.subplots(
                figsize=(10, 7)
            )

            ax.scatter(
                quantum_df[
                    "longitude"
                ],
                quantum_df[
                    "latitude"
                ],
                s=70,
                label="Candidates"
            )

            ax.scatter(
                selected[
                    "longitude"
                ],
                selected[
                    "latitude"
                ],
                s=250,
                marker="*",
                label="Quantum Sensors"
            )

            ax.set_xlabel(
                "Longitude"
            )

            ax.set_ylabel(
                "Latitude"
            )

            ax.set_title(
                "⚛️ Quantum-Optimized Sensor Locations"
            )

            ax.legend()

            ax.grid(
                alpha=0.3
            )

            st.pyplot(fig)

            # ------------------------------------------------
            # SAVE RESULTS
            # ------------------------------------------------

            os.makedirs(
                "results",
                exist_ok=True
            )

            selected.to_csv(
                "results/quantum_sensor_locations.csv",
                index=False
            )

            st.success(
                "Results saved to "
                "`results/quantum_sensor_locations.csv`"
            )


# ============================================================
# DISASTER RESPONSE
# ============================================================

elif page == "🚨 Disaster Response":

    st.header(
        "🚨 Disaster Response Recommendations"
    )

    high_risk_locations = (
        prediction_df[
            prediction_df[
                "flood_probability"
            ] >= 0.50
        ]
        .sort_values(
            "flood_probability",
            ascending=False
        )
    )

    very_high_locations = (
        prediction_df[
            prediction_df[
                "flood_probability"
            ] >= 0.75
        ]
        .sort_values(
            "flood_probability",
            ascending=False
        )
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "High Risk Areas",
        len(high_risk_locations)
    )

    col2.metric(
        "Very High Risk Areas",
        len(very_high_locations)
    )

    total_population = (
        high_risk_locations[
            "population"
        ].sum()
    )

    col3.metric(
        "Population at Risk",
        f"{int(total_population):,}"
    )

    st.divider()

    st.subheader(
        "🚨 Recommended Actions"
    )

    if len(
        very_high_locations
    ) > 0:

        st.error(
            """
            VERY HIGH RISK detected.

            Recommended actions:

            1. Activate emergency response teams.
            2. Issue flood warnings.
            3. Prepare evacuation routes.
            4. Move vulnerable populations to safe areas.
            5. Monitor river levels continuously.
            6. Ensure emergency communication nodes are active.
            """
        )

    elif len(
        high_risk_locations
    ) > 0:

        st.warning(
            """
            HIGH FLOOD RISK detected.

            Recommended actions:

            1. Increase monitoring frequency.
            2. Alert local disaster management teams.
            3. Prepare evacuation resources.
            4. Monitor rainfall and river-level changes.
            5. Check communication infrastructure.
            """
        )

    else:

        st.success(
            """
            Current predicted flood risk is relatively low.

            Continue normal monitoring and maintain
            disaster-response readiness.
            """
        )

    st.subheader(
        "Critical Locations"
    )

    if len(
        high_risk_locations
    ) > 0:

        st.dataframe(
            high_risk_locations[
                [
                    "latitude",
                    "longitude",
                    "flood_probability",
                    "predicted_risk",
                    "river_level",
                    "rainfall_24h",
                    "population"
                ]
            ].head(20),
            use_container_width=True
        )

    st.subheader(
        "📡 Communication Node Recommendation"
    )

    st.info(
        """
        Communication nodes should be prioritized near:

        • Very-high flood-risk locations

        • High population areas

        • Important evacuation routes

        • Areas where existing communication coverage
          may be vulnerable during flooding

        • Major sensor clusters
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.sidebar.divider()

st.sidebar.success(
    "🌊 AI + Quantum Flood Management"
)

st.sidebar.caption(
    "Krishna-Godavari Basin Project"
)