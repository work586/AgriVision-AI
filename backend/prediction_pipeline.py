from pathlib import Path
import sys
import time


# ============================================================
# PROJECT PATH
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ============================================================
# IMPORT PROJECT MODULES
# ============================================================

from backend.image_prediction import (
    predict_crop,
    predict_disease,
)

from backend.price_prediction import (
    get_latest_price,
)

from backend.database import (
    save_analysis,
)


# ============================================================
# COMPLETE IMAGE ANALYSIS PIPELINE
# ============================================================

def analyze_image(image_path, save=True):
    """
    Complete AgriVision AI analysis pipeline.

    Pipeline:

        Image
          ↓
        Image quality check
          ↓
        Crop prediction
          ↓
        Confidence / uncertainty check
          ↓
        Disease prediction
          ↓
        Market price lookup
          ↓
        Save result
          ↓
        Return result

    If crop recognition is unreliable:

        Image
          ↓
        Crop prediction
          ↓
        UNCERTAIN
          ↓
        STOP

    This prevents unreliable crop predictions from being
    passed to the disease and market-price stages.
    """

    timings = {}


    # ========================================================
    # STEP 1 — CROP PREDICTION
    # ========================================================

    _t0 = time.perf_counter()


    crop_result = predict_crop(
        image_path
    )


    timings[
        "crop_prediction"
    ] = (
        time.perf_counter()
        - _t0
    )


    # ========================================================
    # EXTRACT CROP INFORMATION
    # ========================================================

    crop = crop_result.get(
        "crop"
    )


    crop_confidence = (
        crop_result.get(
            "crop_confidence"
        )
    )


    top_crop_predictions = (
        crop_result.get(
            "top_crop_predictions",
            []
        )
    )


    other_crop_predictions = (
        crop_result.get(
            "other_crop_predictions",
            []
        )
    )


    prediction_status = (
        crop_result.get(
            "prediction_status",
            "UNKNOWN"
        )
    )


    prediction_message = (
        crop_result.get(
            "prediction_message"
        )
    )


    prediction_reliable = (
        crop_result.get(
            "prediction_reliable",
            False
        )
    )


    confidence_margin = (
        crop_result.get(
            "confidence_margin"
        )
    )


    # ========================================================
    # STEP 2 — BLOCK UNCERTAIN IMAGE
    # ========================================================

    if not prediction_reliable:

        result = {

            # ------------------------------------------------
            # Crop
            # ------------------------------------------------

            "crop":
                crop,

            "crop_confidence":
                crop_confidence,

            "top_crop_predictions":
                top_crop_predictions,

            "other_crop_predictions":
                other_crop_predictions,

            # ------------------------------------------------
            # Prediction status
            # ------------------------------------------------

            "prediction_status":
                prediction_status,

            "prediction_message":
                prediction_message,

            "prediction_reliable":
                False,

            "confidence_margin":
                confidence_margin,

            # ------------------------------------------------
            # Disease blocked
            # ------------------------------------------------

            "disease":
                "Disease analysis not performed",

            "disease_confidence":
                None,

            "status":
                "UNCERTAIN",

            "disease_model_available":
                False,

            "disease_model_source":
                None,

            # ------------------------------------------------
            # Price blocked
            # ------------------------------------------------

            "price_available":
                False,

            "price_date":
                None,

            "min_price":
                None,

            "max_price":
                None,

            "avg_modal_price":
                None,

            "market_records":
                [],

            # ------------------------------------------------
            # Pipeline status
            # ------------------------------------------------

            "analysis_blocked":
                True,

            "analysis_block_reason":
                prediction_message,
        }


        # ====================================================
        # SAVE BLOCKED RESULT
        # ====================================================

        if save:

            _t0 = time.perf_counter()


            try:

                analysis_id = (
                    save_analysis(
                        result
                    )
                )


                result[
                    "analysis_id"
                ] = analysis_id


                result[
                    "database_saved"
                ] = True


            except Exception as exc:

                result[
                    "analysis_id"
                ] = None


                result[
                    "database_saved"
                ] = False


                result[
                    "database_error"
                ] = str(exc)


            timings[
                "save_to_db"
            ] = (
                time.perf_counter()
                - _t0
            )


        result[
            "timings"
        ] = timings


        return result


    # ========================================================
    # STEP 3 — DISEASE PREDICTION
    # ========================================================

    _t0 = time.perf_counter()


    disease_result = predict_disease(
        image_path,
        crop
    )


    timings[
        "disease_prediction"
    ] = (
        time.perf_counter()
        - _t0
    )


    disease = disease_result.get(
        "disease",
        "Disease model unavailable"
    )


    disease_confidence = (
        disease_result.get(
            "disease_confidence"
        )
    )


    status = disease_result.get(
        "status",
        "UNKNOWN"
    )


    disease_model_available = (
        disease_result.get(
            "disease_model_available",
            False
        )
    )


    disease_model_source = (
        disease_result.get(
            "disease_model_source"
        )
    )


    # ========================================================
    # STEP 4 — MARKET PRICE
    # ========================================================

    _t0 = time.perf_counter()


    try:

        price_result = (
            get_latest_price(
                crop
            )
        )


    except Exception as exc:

        price_result = {

            "available":
                False,

            "crop":
                crop,

            "message":
                str(exc),
        }


    timings[
        "price_lookup"
    ] = (
        time.perf_counter()
        - _t0
    )


    price_available = (
        price_result.get(
            "available",
            False
        )
    )


    price_date = (
        price_result.get(
            "date"
        )
    )


    min_price = (
        price_result.get(
            "min_price"
        )
    )


    max_price = (
        price_result.get(
            "max_price"
        )
    )


    avg_modal_price = (
        price_result.get(
            "modal_price"
        )
    )


    market_records = (
        price_result.get(
            "records",
            []
        )
    )


    # ========================================================
    # STEP 5 — BUILD RESULT
    # ========================================================

    result = {

        # ----------------------------------------------------
        # Crop
        # ----------------------------------------------------

        "crop":
            crop,

        "crop_confidence":
            crop_confidence,

        "top_crop_predictions":
            top_crop_predictions,

        "other_crop_predictions":
            other_crop_predictions,

        "prediction_status":
            prediction_status,

        "prediction_message":
            prediction_message,

        "prediction_reliable":
            True,

        "confidence_margin":
            confidence_margin,

        # ----------------------------------------------------
        # Disease
        # ----------------------------------------------------

        "disease":
            disease,

        "disease_confidence":
            disease_confidence,

        "status":
            status,

        "disease_model_available":
            disease_model_available,

        "disease_model_source":
            disease_model_source,

        # ----------------------------------------------------
        # Market
        # ----------------------------------------------------

        "price_available":
            price_available,

        "price_date":
            price_date,

        "min_price":
            min_price,

        "max_price":
            max_price,

        "avg_modal_price":
            avg_modal_price,

        "market_records":
            market_records,

        # ----------------------------------------------------
        # Pipeline
        # ----------------------------------------------------

        "analysis_blocked":
            False,

        "analysis_block_reason":
            None,
    }


    # ========================================================
    # STEP 6 — SAVE TO DATABASE
    # ========================================================

    if save:

        _t0 = time.perf_counter()


        try:

            analysis_id = (
                save_analysis(
                    result
                )
            )


            result[
                "analysis_id"
            ] = analysis_id


            result[
                "database_saved"
            ] = True


        except Exception as exc:

            result[
                "analysis_id"
            ] = None


            result[
                "database_saved"
            ] = False


            result[
                "database_error"
            ] = str(exc)


        timings[
            "save_to_db"
        ] = (
            time.perf_counter()
            - _t0
        )


    # ========================================================
    # RETURN RESULT
    # ========================================================

    result[
        "timings"
    ] = timings


    return result


# ============================================================
# COMMAND-LINE INFORMATION
# ============================================================

if __name__ == "__main__":

    print()

    print("=" * 60)

    print(
        "AgriVision AI — Prediction Pipeline"
    )

    print("=" * 60)

    print()

    print(
        "Image quality validation : Enabled"
    )

    print(
        "Crop prediction          : Enabled"
    )

    print(
        "Top-3 predictions        : Enabled"
    )

    print(
        "Confidence gate          : Enabled"
    )

    print(
        "Disease prediction       : Enabled"
    )

    print(
        "Market price lookup      : Enabled"
    )

    print(
        "SQLite database           : Enabled"
    )

    print()

    print(
        "This module is normally called "
        "by the Streamlit app."
    )

    print()