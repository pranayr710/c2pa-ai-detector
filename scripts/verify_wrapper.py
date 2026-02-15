#!/usr/bin/env python3
"""
Image Verification Wrapper
Simple, robust script that outputs JSON only
"""

import sys
import json
import os

def main():
    try:
        if len(sys.argv) < 2:
            print(json.dumps({
                'has_seal': False,
                'is_valid': False,
                'source_type': 'UNKNOWN',
                'confidence': 0.0,
                'verification_method': 'none',
                'details': {},
                'error': 'No file path provided'
            }))
            return
        
        file_path = sys.argv[1]
        
        if not os.path.exists(file_path):
            print(json.dumps({
                'has_seal': False,
                'is_valid': False,
                'source_type': 'UNKNOWN',
                'confidence': 0.0,
                'verification_method': 'none',
                'details': {},
                'error': f'File not found: {file_path}'
            }))
            return
        
        # Add current directory to path
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Check if video
        video_exts = ('.mp4', '.mov', '.avi', '.mkv', '.webm')
        if file_path.lower().endswith(video_exts):
            from video_detector import process_video
            
            # Run video verification
            result = process_video(file_path)
            
            if "error" in result:
                raise Exception(result["error"])
                
            # Format result to match frontend expectations
            response = {
                'has_seal': False, # Videos don't have C2PA seals in this MVP
                'is_valid': not result['is_fake'],
                'source_type': 'VIDEO',
                'confidence': result['confidence'],
                'verification_method': 'hybrid_fft_video',
                'details': {
                    'fake_ratio': result['fake_ratio'],
                    'frames_analyzed': result['frames_analyzed'],
                    'fake_frames_count': result['fake_frames_count']
                }
            }
            print(json.dumps(response))
            return

        # Fallback to existing Image Verification
        from seal_verifier import SealVerifier
        
        verifier = SealVerifier()
        result = verifier.verify_image(file_path)
        
        # Output result as JSON
        print(json.dumps(result.to_dict()))
        
    except Exception as e:
        # Always output valid JSON, even on error
        error_msg = str(e).replace('"', "'")
        print(json.dumps({
            'has_seal': False,
            'is_valid': False,
            'source_type': 'UNKNOWN',
            'confidence': 0.0,
            'verification_method': 'none',
            'details': {},
            'error': error_msg
        }))

if __name__ == '__main__':
    main()
