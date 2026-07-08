"""
PayScan AI Agent - Duplicate Detection Function
Detects duplicate invoices
"""

import json
import boto3
import os
import logging
from typing import Dict, Any, List
from datetime import datetime

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3_client = boto3.client('s3')

class PayScanDuplicateDetector:
    def __init__(self):
        self.results_bucket = os.environ.get('RESULTS_BUCKET', 'payscan-results-prod')
        self.amount_tolerance = 0.01  # $0.01 tolerance for amount matching
    
    def lambda_handler(self, event: Dict[str, Any], context: Any) -> Dict[str, Any]:
        """Main Lambda handler for duplicate detection"""
        try:
            logger.info(f"Starting duplicate check: {json.dumps(event, default=str)}")
            
            job_id = event['job_id']
            invoice_data = event['invoice_data']
            
            current_invoice_id = invoice_data.get('invoice_id')
            current_amount = invoice_data.get('amount')
            current_vendor = invoice_data.get('vendor_name')
            
            duplicates_found = []
            
            # Only check if we have key matching data
            if current_invoice_id or (current_amount and current_vendor):
                duplicates_found = self.find_duplicates(
                    current_invoice_id,
                    current_amount,
                    current_vendor,
                    job_id
                )
            
            return {
                'success': True,
                'job_id': job_id,
                'duplicates_found': duplicates_found,
                'duplicate_count': len(duplicates_found),
                'has_duplicates': len(duplicates_found) > 0,
                'check_timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error checking duplicates: {str(e)}")
            return {
                'success': False,
                'job_id': event.get('job_id'),
                'error': str(e),
                'error_type': 'duplicate_detection_error'
            }
    
    def find_duplicates(self, invoice_id: str, amount: float, vendor: str, current_job_id: str) -> List[Dict[str, Any]]:
        """Search for duplicate invoices"""
        try:
            duplicates = []
            
            # List all processed invoices
            paginator = s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.results_bucket, Prefix='processed/')
            
            for page in pages:
                if 'Contents' not in page:
                    continue
                
                for obj in page['Contents']:
                    if obj['Key'].endswith('final_result.json') and current_job_id not in obj['Key']:
                        try:
                            duplicate = self.check_invoice_match(obj['Key'], invoice_id, amount, vendor)
                            if duplicate:
                                duplicates.append(duplicate)
                        except Exception as e:
                            logger.warning(f"Error checking {obj['Key']}: {str(e)}")
                            continue
            
            return duplicates
            
        except Exception as e:
            logger.error(f"Error finding duplicates: {str(e)}")
            return []
    
    def check_invoice_match(self, s3_key: str, invoice_id: str, amount: float, vendor: str) -> Dict[str, Any] or None:
        """Check if invoice matches current invoice"""
        try:
            # Get existing invoice data
            response = s3_client.get_object(Bucket=self.results_bucket, Key=s3_key)
            existing_data = json.loads(response['Body'].read())
            
            quick_access = existing_data.get('quick_access', {})
            existing_invoice_id = quick_access.get('invoice_id')
            existing_amount = quick_access.get('amount')
            existing_vendor = quick_access.get('vendor_name')
            existing_job_id = existing_data.get('job_id')
            
            match_reasons = []
            confidence = 0.0
            
            # Check invoice ID exact match (highest confidence)
            if invoice_id and existing_invoice_id:
                if invoice_id.lower().strip() == existing_invoice_id.lower().strip():
                    match_reasons.append('invoice_id_exact_match')
                    confidence = 0.99
            
            # Check amount + vendor match (high confidence)
            if amount and existing_amount and vendor and existing_vendor:
                amount_match = abs(float(amount) - float(existing_amount)) <= self.amount_tolerance
                vendor_match = vendor.lower().strip() == existing_vendor.lower().strip()
                
                if amount_match and vendor_match:
                    if 'invoice_id_exact_match' not in match_reasons:
                        match_reasons.append('amount_vendor_match')
                        confidence = 0.95
            
            # Only return if we found matches
            if match_reasons:
                return {
                    'existing_job_id': existing_job_id,
                    'existing_invoice_id': existing_invoice_id,
                    'existing_amount': existing_amount,
                    'existing_vendor': existing_vendor,
                    'match_reasons': match_reasons,
                    'confidence': confidence,
                    'storage_location': f"s3://{self.results_bucket}/{s3_key}",
                    'detected_at': datetime.utcnow().isoformat()
                }
            
            return None
            
        except Exception as e:
            logger.warning(f"Error checking match: {str(e)}")
            return None

detector = PayScanDuplicateDetector()

def lambda_handler(event, context):
    """Lambda entry point"""
    return detector.lambda_handler(event, context)
