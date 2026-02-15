"""
Unified Seal Verifier (Lightweight Provenance + ML Fallback)

Workflow:
1. Detect lightweight seal (PNG metadata / JPG EXIF)
2. Verify seal integrity using pixel hash
3. If seal valid → VERIFIED_REAL
4. If seal tampered → TAMPERED + ML fallback
5. If no seal → ML detection only
"""

from typing import Optional, Dict, Any
from dataclasses import dataclass, field
import os

from PIL import Image
import piexif

from utils import get_pixel_hash, get_c2pa_seal


# ==========================================================
# Result Dataclass
# ==========================================================
@dataclass
class VerificationResult:
    has_seal: bool = False
    is_valid: bool = False
    source_type: str = "UNKNOWN"   # VERIFIED_REAL, TAMPERED, LIKELY_REAL, LIKELY_AI
    source_name: Optional[str] = None
    confidence: float = 0.0
    verification_method: str = "unknown"
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self):
        return {
            "has_seal": self.has_seal,
            "is_valid": self.is_valid,
            "source_type": self.source_type,
            "source_name": self.source_name,
            "confidence": self.confidence,
            "verification_method": self.verification_method,
            "details": self.details,
            "error": self.error
        }


# ==========================================================
# Seal Verifier
# ==========================================================
class SealVerifier:
    """Unified verification with seal priority and ML fallback"""

    def __init__(self):
        try:
            from ml_detector_wrapper import detect_ai_likelihood
            self.detect_ai = detect_ai_likelihood
            self.has_ml = True
        except Exception:
            self.has_ml = False

    # ------------------------------------------------------
    def verify_image(self, image_path: str) -> VerificationResult:
        result = VerificationResult()

        # -------------------------------
        # File check
        # -------------------------------
        if not os.path.exists(image_path):
            result.error = "Image file not found"
            return result

        # -------------------------------
        # STEP 1: Seal detection
        # -------------------------------
        seal_hash = get_c2pa_seal(image_path)

        if seal_hash:
            result.has_seal = True

            try:
                current_hash = get_pixel_hash(image_path)
            except Exception as e:
                result.error = str(e)
                return result

            # -------------------------------
            # SEAL VALID
            # -------------------------------
            if seal_hash == current_hash:
                result.is_valid = True
                result.source_type = "VERIFIED_REAL"
                result.source_name = "Lightweight Provenance Seal"
                result.confidence = 1.0
                result.verification_method = "C2PA_SEAL"
                return result

            # -------------------------------
            # SEAL TAMPERED → ML FALLBACK
            # -------------------------------
            result.is_valid = False
            result.source_type = "TAMPERED"
            result.verification_method = "C2PA_SEAL"
            result.details["reason"] = "Seal mismatch"

            if self.has_ml:
                try:
                    is_ai, confidence = self.detect_ai(image_path)
                    result.details["ml_verdict"] = "LIKELY_AI" if is_ai else "LIKELY_REAL"
                    result.details["ml_confidence"] = confidence
                except Exception as e:
                    result.details["ml_error"] = str(e)

            return result

        # -------------------------------
        # STEP 2: No seal → ML only
        # -------------------------------
        if self.has_ml:
            try:
                is_ai, confidence = self.detect_ai(image_path)
                result.source_type = "LIKELY_AI" if is_ai else "LIKELY_REAL"
                result.confidence = confidence
                result.verification_method = "ML_DETECTION"
                return result
            except Exception as e:
                result.error = str(e)
                return result

        result.error = "No verification method available"
        return result



















# """
# Unified Seal Verifier
# Orchestrates the complete verification workflow:
# 1. Detect C2PA seal
# 2. Parse manifest
# 3. Verify cryptographic signature and hash
# 4. Validate trust source
# 5. Fallback to ML detection if seal not found/invalid
# """

# from typing import Tuple, Optional, Dict, Any
# from dataclasses import dataclass, field
# import os
# import sys
# import traceback


# @dataclass
# class VerificationResult:
#     """Result of seal verification"""
#     has_seal: bool = False
#     is_valid: bool = False
#     source_type: str = "UNKNOWN"  # 'VERIFIED_REAL', 'VERIFIED_AI', 'LIKELY_AI', 'LIKELY_REAL', 'UNKNOWN'
#     source_name: Optional[str] = None
#     confidence: float = 0.0
#     verification_method: str = "unknown"
#     details: Dict[str, Any] = field(default_factory=dict)
#     error: Optional[str] = None
    
#     def to_dict(self) -> Dict[str, Any]:
#         return {
#             'has_seal': self.has_seal,
#             'is_valid': self.is_valid,
#             'source_type': self.source_type,
#             'source_name': self.source_name,
#             'confidence': self.confidence,
#             'verification_method': self.verification_method,
#             'details': self.details,
#             'error': self.error
#         }


# class SealVerifier:
#     """Complete seal verification workflow with ML fallback"""
    
#     def __init__(self):
#         self.has_seal_parser = False
#         self.has_crypto = False
#         self.has_registry = False
#         self.has_ml = False
        
#         try:
#             from c2pa_seal_parser import C2PASealDetector
#             self.detector = C2PASealDetector()
#             self.has_seal_parser = True
#         except Exception as e:
#             pass
        
#         try:
#             from crypto_verifier import CryptoVerifier
#             self.crypto = CryptoVerifier()
#             self.has_crypto = True
#         except Exception as e:
#             pass
        
#         try:
#             from trust_registry import TrustRegistry
#             self.trust_registry = TrustRegistry()
#             self.has_registry = True
#         except Exception as e:
#             pass
        
#         try:
#             from ml_detector_wrapper import detect_ai_likelihood
#             self.detect_ai = detect_ai_likelihood
#             self.has_ml = True
#         except Exception as e:
#             pass
    
#     def verify_image(self, image_path: str) -> VerificationResult:
#         """Main verification workflow"""
        
#         if not os.path.exists(image_path):
#             return VerificationResult(
#                 error=f"Image file not found: {image_path}"
#             )
        
#         result = VerificationResult()
        
#         try:
#             # Step 1: Try to detect C2PA seal
#             if self.has_seal_parser:
#                 try:
#                     seal_data = self.detector.detect_seal(image_path)
#                     if seal_data:
#                         result.has_seal = True
#                         # Step 2: Verify crypto signature
#                         if self.has_crypto and self.has_registry:
#                             is_valid, source_info = self._verify_seal_crypto(seal_data)
#                             result.is_valid = is_valid
#                             result.source_type = source_info.get('type', 'UNKNOWN')
#                             result.source_name = source_info.get('name')
#                             result.confidence = source_info.get('confidence', 1.0 if is_valid else 0.0)
#                             result.verification_method = 'crypto'
#                             result.details = source_info.get('details', {})
#                             return result
#                 except Exception as e:
#                     pass
            
#             # Step 3: Fallback to ML detection
#             if self.has_ml:
#                 try:
#                     ai_likelihood, confidence = self.detect_ai(image_path)
#                     result.source_type = 'LIKELY_AI' if ai_likelihood else 'LIKELY_REAL'
#                     result.confidence = confidence
#                     result.verification_method = 'ml'
#                     result.details = {'method': 'neural_network'}
#                     return result
#                 except Exception as e:
#                     result.error = f"ML detection failed: {str(e)}"
#                     return result
            
#             # No verification method available
#             result.error = "No verification method available"
#             return result
            
#         except Exception as e:
#             result.error = f"Verification failed: {str(e)}"
#             return result
    
#     def _verify_seal_crypto(self, seal_data: Dict[str, Any]) -> tuple:
#         """Verify cryptographic seal"""
#         try:
#             # Verify signature
#             is_valid = self.crypto.verify_signature(seal_data)
            
#             # Get source from trust registry
#             source_name = seal_data.get('source', {}).get('name', 'Unknown')
#             source_type = self.trust_registry.get_source_type(source_name)
            
#             source_info = {
#                 'type': source_type,
#                 'name': source_name,
#                 'confidence': 1.0 if is_valid else 0.0,
#                 'details': {
#                     'signature_valid': is_valid,
#                     'timestamp': seal_data.get('timestamp'),
#                 }
#             }
            
#             return is_valid, source_info
#         except Exception as e:
#             return False, {'type': 'UNKNOWN', 'error': str(e), 'details': {}}


