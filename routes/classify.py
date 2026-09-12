"""Image classification routes for NSFW and aesthetic scoring."""

import base64
import logging
from io import BytesIO

from flask import Blueprint, request, jsonify
from PIL import Image

from ..classifiers import classifier_registry

logger = logging.getLogger(__name__)

# Blueprint with /classify prefix
classify_bp = Blueprint("classify", __name__, url_prefix="/classify")

# Blueprint for direct root-level routes (/nsfw, /aesthetic)
classify_root_bp = Blueprint("classify_root", __name__)


def _get_image_from_request() -> Image.Image:
    """
    Extract image from request - supports multiple formats:
    1. Raw bytes with Content-Type: application/octet-stream or image/*
    2. JSON with base64-encoded 'image' field
    3. Form data with 'image' file field
    """
    content_type = request.content_type or ""
    
    # Raw bytes (application/octet-stream or image/*)
    if content_type.startswith("application/octet-stream") or content_type.startswith("image/"):
        image_bytes = request.get_data()
        if not image_bytes:
            raise ValueError("No image data in request body")
        return Image.open(BytesIO(image_bytes))
    
    # JSON with base64 image
    if content_type.startswith("application/json"):
        data = request.get_json() or {}
        if "image" in data:
            image_bytes = base64.b64decode(data["image"])
            return Image.open(BytesIO(image_bytes))
        raise ValueError("JSON request must include 'image' field with base64 data")
    
    # Form data
    if "multipart/form-data" in content_type:
        if "image" in request.files:
            return Image.open(request.files["image"])
        raise ValueError("Form data must include 'image' file field")
    
    # Try to read raw data as fallback
    image_bytes = request.get_data()
    if image_bytes:
        try:
            return Image.open(BytesIO(image_bytes))
        except Exception:
            pass
    
    raise ValueError(
        "Unsupported request format. Send image as: "
        "1) Raw bytes with Content-Type: application/octet-stream, "
        "2) JSON with base64 'image' field, or "
        "3) Form data with 'image' file"
    )


def _get_threshold_from_request() -> float:
    """Get threshold from request (JSON body or query param)."""
    # Try JSON body first
    if request.content_type and request.content_type.startswith("application/json"):
        data = request.get_json() or {}
        if "threshold" in data:
            return float(data["threshold"])
    
    # Try query param
    threshold = request.args.get("threshold")
    if threshold:
        return float(threshold)
    
    return 0.8  # Default


def _handle_nsfw_request() -> dict:
    """Handle NSFW classification request (any format)."""
    image = _get_image_from_request()
    threshold = _get_threshold_from_request()
    
    classifier = classifier_registry.get("nsfw")
    if hasattr(classifier, 'threshold'):
        classifier.threshold = float(threshold)
    
    result = classifier.predict(image)
    
    return {
        "is_nsfw": result.raw.get("is_nsfw", result.label == "nsfw"),
        "score": result.score,
        "label": result.label,
        "threshold": threshold,
    }


def _handle_aesthetic_request() -> dict:
    """Handle aesthetic classification request (any format)."""
    image = _get_image_from_request()
    
    classifier = classifier_registry.get("aesthetic")
    result = classifier.predict(image)
    
    return {
        "score": result.score,
        "label": result.label,
        "quality_tier": result.raw.get("quality_tier", result.label),
        "scale": "1-10",
    }


def _get_age_threshold_from_request() -> int:
    """Get adult age threshold from request (JSON body or query param)."""
    if request.content_type and request.content_type.startswith("application/json"):
        data = request.get_json() or {}
        if "adult_threshold" in data:
            return int(data["adult_threshold"])
    
    threshold = request.args.get("adult_threshold")
    if threshold:
        return int(threshold)
    
    return 18  # Default


def _get_url_from_request() -> str:
    """Get image URL from request (JSON body, query param, or plain text body)."""
    # JSON body with url field
    if request.content_type and request.content_type.startswith("application/json"):
        data = request.get_json() or {}
        if "url" in data:
            return data["url"]
    
    # Query param
    url = request.args.get("url")
    if url:
        return url
    
    # Plain text body (just the URL)
    if request.content_type and request.content_type.startswith("text/plain"):
        body = request.get_data(as_text=True).strip()
        if body.startswith("http"):
            return body
    
    return None


def _handle_deepface_age_request() -> dict:
    """Handle DeepFace age classification request."""
    classifier = classifier_registry.get("deepface_age")
    threshold = _get_age_threshold_from_request()
    
    # Check for URL-based request first
    url = _get_url_from_request()
    if url:
        result = classifier.predict_from_url(url)
    else:
        image = _get_image_from_request()
        result = classifier.predict(image)
    
    # Check for errors
    if result.raw.get("error"):
        return {
            "error": result.raw["error"],
            "estimated_age": None,
            "is_adult": None,
            "faces": []
        }
    
    # Apply custom threshold if different from default
    est_age = result.raw.get("estimated_age")
    is_adult = est_age >= threshold if est_age is not None else None
    
    return {
        "estimated_age": est_age,
        "is_adult": is_adult,
        "adult_threshold": threshold,
        "label": result.label,
        "face_confidence": result.raw.get("face_confidence", 0),
        "bounding_box": result.raw.get("bounding_box"),
        "faces": result.raw.get("all_faces", [])
    }


# ============ Routes under /classify/* ============

@classify_bp.route("/nsfw", methods=["POST"])
def classify_nsfw():
    """Detect NSFW content (via /classify/nsfw)."""
    try:
        return jsonify(_handle_nsfw_request())
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.exception("NSFW classification failed")
        return jsonify({"error": str(e)}), 500


@classify_bp.route("/aesthetic", methods=["POST"])
def classify_aesthetic():
    """Score aesthetic quality (via /classify/aesthetic)."""
    try:
        return jsonify(_handle_aesthetic_request())
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.exception("Aesthetic classification failed")
        return jsonify({"error": str(e)}), 500


@classify_bp.route("/age", methods=["POST"])
def classify_deepface_age():
    """
    Estimate age using DeepFace (via /classify/age).
    
    Accepts:
        - Raw image bytes with Content-Type: application/octet-stream
        - JSON with base64 'image' field OR 'url' field for URL-based detection
        - Form data with 'image' file
        - Query param ?url=<image_url> for URL-based detection
        - Query param ?adult_threshold=<age> to customize adult threshold (default 18)
    
    Returns:
        JSON with estimated_age, is_adult, face_confidence, bounding_box, faces (all detected)
    """
    try:
        return jsonify(_handle_deepface_age_request())
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.exception("DeepFace age classification failed")
        return jsonify({"error": str(e)}), 500


@classify_bp.route("/status", methods=["GET"])
def classifier_status():
    """Get classifier system status."""
    return jsonify({
        "current": classifier_registry.current_name,
        "available": classifier_registry.available,
    })


@classify_bp.route("/unload", methods=["POST"])
def unload_classifier():
    """Unload the current classifier to free memory."""
    try:
        current = classifier_registry.current_name
        classifier_registry.unload_current()
        return jsonify({
            "success": True,
            "unloaded": current,
            "message": f"Classifier '{current}' unloaded" if current else "No classifier was loaded",
        })
    except Exception as e:
        logger.exception("Failed to unload classifier")
        return jsonify({"error": str(e)}), 500


# ============ Direct routes at root (/nsfw, /aesthetic) ============

@classify_root_bp.route("/nsfw", methods=["POST"])
def nsfw_direct():
    """
    Detect NSFW content in an image.
    
    Accepts:
        - Raw image bytes with Content-Type: application/octet-stream
        - JSON with base64-encoded 'image' field
        - Form data with 'image' file
    
    Returns:
        JSON with is_nsfw (bool), score (float), and label.
    """
    try:
        return jsonify(_handle_nsfw_request())
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.exception("NSFW classification failed")
        return jsonify({"error": str(e)}), 500


@classify_root_bp.route("/aesthetic", methods=["POST"])
def aesthetic_direct():
    """
    Score image aesthetic quality on 1-10 scale.
    
    Accepts:
        - Raw image bytes with Content-Type: application/octet-stream
        - JSON with base64-encoded 'image' field
        - Form data with 'image' file
    
    Returns:
        JSON with score (1-10), label (quality tier), and details.
    """
    try:
        return jsonify(_handle_aesthetic_request())
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.exception("Aesthetic classification failed")
        return jsonify({"error": str(e)}), 500


@classify_root_bp.route("/age", methods=["POST"])
def age_direct():
    """
    Estimate age using DeepFace with RetinaFace detector.
    
    Accepts:
        - Raw image bytes with Content-Type: application/octet-stream
        - JSON with base64 'image' field OR 'url' field
        - Form data with 'image' file
        - Query params: ?url=<url>&adult_threshold=<age>
    
    Returns:
        JSON with estimated_age, is_adult, face_confidence, bounding_box, faces
    """
    try:
        return jsonify(_handle_deepface_age_request())
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.exception("DeepFace age classification failed")
        return jsonify({"error": str(e)}), 500
