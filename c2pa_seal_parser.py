"""
C2PA (Coalition for Content Provenance and Authenticity) Seal Parser
Detects, extracts, and parses C2PA manifests embedded in images
"""

import json
import struct
from typing import Optional, Dict, Any
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import io


class C2PAManifest:
    """Represents a C2PA manifest extracted from an image"""
    
    def __init__(self, manifest_data: Dict[str, Any]):
        self.data = manifest_data
        self.title = manifest_data.get('title', 'Unknown')
        self.claim_generator = manifest_data.get('claim_generator', {})
        self.assertions = manifest_data.get('assertions', [])
        self.signature_block = manifest_data.get('signature', {})
        
    def get_source(self) -> Optional[str]:
        """Extract the source/creator from assertions"""
        for assertion in self.assertions:
            if 'c2pa.actions' in assertion:
                action_data = assertion.get('data', {})
                return action_data.get('source', {}).get('name')
        return None
    
    def get_creation_time(self) -> Optional[str]:
        """Extract creation timestamp"""
        for assertion in self.assertions:
            if 'c2pa.created' in assertion:
                return assertion.get('data')
        return None
    
    def get_all_assertions(self) -> list:
        """Return all assertions in the manifest"""
        return self.assertions


class C2PASealDetector:
    """Detects and extracts C2PA seals from images"""
    
    # C2PA JUMBF markers
    C2PA_MARKER = b'jumb'
    C2PA_UUID = bytes.fromhex('6d6e6c6546202e32000000001000000')  # C2PA UUID
    
    @staticmethod
    def has_c2pa_seal(image_path: str) -> bool:
        """Check if an image has a C2PA seal"""
        try:
            img = Image.open(image_path)
            
            # Check PNG chunks
            if img.format == 'PNG':
                if hasattr(img, 'info'):
                    for key in img.info:
                        if key.lower() == 'c2pa' or 'jumb' in str(key).lower():
                            return True
            
            # Check EXIF/metadata
            if hasattr(img, '_getexif') and img._getexif():
                exif = img._getexif()
                if exif and any('c2pa' in str(v).lower() for v in exif.values()):
                    return True
            
            # Check for embedded JSON
            with open(image_path, 'rb') as f:
                data = f.read()
                if b'c2pa' in data.lower() or b'jumb' in data:
                    return True
                    
        except Exception as e:
            print(f"[v0] Error checking C2PA seal: {e}")
            return False
        
        return False
    
    @staticmethod
    def extract_seal(image_path: str) -> Optional[C2PAManifest]:
        """Extract C2PA manifest from image"""
        try:
            with open(image_path, 'rb') as f:
                data = f.read()
            
            # Look for C2PA JSON manifest in file
            # C2PA typically stores manifest as JSON
            start_idx = data.find(b'"claim_generator":')
            if start_idx == -1:
                start_idx = data.find(b'"title":')
            
            if start_idx != -1:
                # Find the start of the JSON object
                json_start = data.rfind(b'{', 0, start_idx)
                if json_start == -1:
                    return None
                
                # Find the end of JSON
                json_end = data.find(b'}}', json_start) + 2
                
                manifest_json = data[json_start:json_end].decode('utf-8', errors='ignore')
                manifest_dict = json.loads(manifest_json)
                
                return C2PAManifest(manifest_dict)
        
        except json.JSONDecodeError as e:
            print(f"[v0] JSON decode error: {e}")
            return None
        except Exception as e:
            print(f"[v0] Error extracting seal: {e}")
            return None
        
        return None
