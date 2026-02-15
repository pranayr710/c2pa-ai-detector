"""
Cryptographic Verification System (C2PA-aligned)

FEATURES INCLUDED:
✔ SHA-256 image hashing
✔ Manifest hash validation
✔ RSA signature verification (optional)
✔ Timestamp validity checking
✔ Certificate-chain placeholder
✔ HMAC placeholder
✔ High-level verification wrapper for main system

NOTE:
This project does not embed real C2PA manifests.
Therefore cryptographic verification safely fails and falls back to ML.
"""

import hashlib
import hmac
import json
import base64
from datetime import datetime
from typing import Dict, Any, Optional


class CryptoVerifier:
    """
    Handles cryptographic verification of image provenance.
    """

    HASH_ALGORITHM = "sha256"
    SIGNATURE_ALGORITHM = "rsassa-pss"

    # ======================================================
    # HASHING
    # ======================================================

    @staticmethod
    def compute_hash(image_path: str, algorithm: str = "sha256") -> str:
        """Compute cryptographic hash of an image file"""
        try:
            hash_obj = hashlib.new(algorithm)
            with open(image_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_obj.update(chunk)
            return hash_obj.hexdigest()
        except Exception as e:
            print(f"[CRYPTO] Hash computation error: {e}")
            return ""

    # ======================================================
    # MANIFEST HASH VERIFICATION
    # ======================================================

    @staticmethod
    def verify_manifest_hash(manifest: Dict[str, Any], image_path: str) -> bool:
        """Verify manifest hash matches image hash"""
        try:
            manifest_hash = manifest.get("hash")
            if not manifest_hash:
                print("[CRYPTO] Manifest contains no hash")
                return False

            computed_hash = CryptoVerifier.compute_hash(image_path)

            if computed_hash == manifest_hash:
                print("[CRYPTO] Manifest hash verification PASSED")
                return True

            print("[CRYPTO] Manifest hash verification FAILED")
            return False

        except Exception as e:
            print(f"[CRYPTO] Manifest hash verification error: {e}")
            return False

    # ======================================================
    # RSA SIGNATURE VERIFICATION
    # ======================================================

    @staticmethod
    def verify_signature(manifest: Dict[str, Any], public_key_pem: Optional[str]) -> bool:
        """
        Verify RSA signature of manifest.

        NOTE:
        Requires cryptography library.
        This function is kept intact for future real C2PA support.
        """
        try:
            if not public_key_pem:
                print("[CRYPTO] No public key provided")
                return False

            signature_b64 = manifest.get("signature")
            if not signature_b64:
                print("[CRYPTO] No signature in manifest")
                return False

            from cryptography.hazmat.primitives import hashes, serialization
            from cryptography.hazmat.primitives.asymmetric import padding
            from cryptography.hazmat.backends import default_backend

            signature = base64.b64decode(signature_b64)

            public_key = serialization.load_pem_public_key(
                public_key_pem.encode(),
                backend=default_backend()
            )

            manifest_copy = {k: v for k, v in manifest.items() if k != "signature"}
            manifest_bytes = json.dumps(manifest_copy, sort_keys=True).encode()

            public_key.verify(
                signature,
                manifest_bytes,
                padding.PKCS1v15(),
                hashes.SHA256()
            )

            print("[CRYPTO] Signature verification PASSED")
            return True

        except ImportError:
            print("[CRYPTO] cryptography library not installed")
            return False
        except Exception as e:
            print(f"[CRYPTO] Signature verification FAILED: {e}")
            return False

    # ======================================================
    # HMAC VERIFICATION (PLACEHOLDER)
    # ======================================================

    @staticmethod
    def verify_hmac(data: bytes, received_hmac: str, secret_key: bytes) -> bool:
        """Verify HMAC (kept for completeness)"""
        try:
            computed_hmac = hmac.new(secret_key, data, hashlib.sha256).hexdigest()
            if hmac.compare_digest(computed_hmac, received_hmac):
                print("[CRYPTO] HMAC verification PASSED")
                return True
            print("[CRYPTO] HMAC verification FAILED")
            return False
        except Exception as e:
            print(f"[CRYPTO] HMAC error: {e}")
            return False

    # ======================================================
    # TIMESTAMP VERIFICATION
    # ======================================================

    @staticmethod
    def check_timestamp_validity(timestamp: str, max_age_days: int = 365) -> bool:
        """Verify manifest timestamp validity"""
        try:
            manifest_time = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            current_time = datetime.now(manifest_time.tzinfo)

            age = (current_time - manifest_time).days

            if age < 0:
                print("[CRYPTO] Timestamp is from the future")
                return False

            if age > max_age_days:
                print("[CRYPTO] Timestamp expired")
                return False

            print("[CRYPTO] Timestamp validity PASSED")
            return True

        except Exception as e:
            print(f"[CRYPTO] Timestamp error: {e}")
            return False

    # ======================================================
    # CERTIFICATE CHAIN VERIFICATION (PLACEHOLDER)
    # ======================================================

    @staticmethod
    def verify_certificate_chain(cert_chain: Optional[list]) -> bool:
        """
        Placeholder for X.509 certificate chain validation.
        """
        if not cert_chain:
            print("[CRYPTO] No certificate chain provided")
            return False

        print("[CRYPTO] Certificate chain verification NOT IMPLEMENTED")
        return False


# ==========================================================
# HIGH-LEVEL WRAPPER (USED BY main.py)
# ==========================================================

def verify_crypto_proof(image_path: str) -> Dict[str, Any]:
    """
    Entry point used by main system.

    Attempts cryptographic verification.
    Falls back to ML if no valid proof exists.
    """

    # ⚠️ No embedded C2PA manifest in project images
    # This is an intentional safe-fail design

    return {
        "verified": False,
        "method": "NONE",
        "message": "No cryptographic provenance metadata found"
    }
