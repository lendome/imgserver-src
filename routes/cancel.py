"""Cancel/abort route for stopping generation."""

from flask import Blueprint, jsonify

from ..abort import abort_controller

cancel_bp = Blueprint("cancel", __name__)


@cancel_bp.route("/cancel", methods=["POST"])
def cancel():
    """Cancel the current generation if one is active."""
    was_active = abort_controller.is_active
    cancelled = abort_controller.abort()
    return jsonify({
        "cancelled": cancelled,
        "was_active": was_active
    })
